import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
import tensorflow_addons as tfa

def get_shape(tensor):
    """
    Deal with dynamic shape in tensorflow cleanly.
    Args:
        tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.
    Returns:
        `List[int]`: The shape of the tensor as a list.
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

class PatchEmbeddings(Layer):
    """Construct the patch embeddings."""
    def __init__(self, num_patches, dims, reshape=True, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.proj = Dense(dims)
        self.pos_embedding = Embedding(input_dim=num_patches, output_dim=dims)
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        
    def call(self, x):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        x = tf.math.reduce_mean(x, axis=-2)
        x = self.proj(x) + self.pos_embedding(positions)
        x = self.layer_norm(x)
        return x

class MultiHeadAttention(Layer):
    """Multi Head Self Attention"""
    def __init__(self, num_heads, dims, dropout_ratio=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.dims = dims
        self.query = Dense(dims, name="query")
        self.key   = Dense(dims, name="key")
        self.value = Dense(dims, name="value")
        self.proj  = Dense(dims, name="proj")
        self.dropout = Dropout(dropout_ratio)

    def call(self, q, x):
        B, P, C = get_shape(x)

        head_dims = self.dims // self.num_heads

        query = self.query(q)
        query = tf.reshape(query, (-1, P, head_dims, self.num_heads))

        key   = self.key(x)
        key   = tf.reshape(key, (-1, P, head_dims, self.num_heads))

        value = self.value(x)
        value = tf.reshape(value, (-1, P, head_dims, self.num_heads))

        heads = []
        for i in range(self.num_heads):
            q, k, v = query[..., i], key[..., i], value[..., i]
            scores  = tf.matmul(q, k, transpose_b=True) / (head_dims ** 0.5) # W: N x N_k, v: N_k x C
            weights = tf.nn.softmax(scores, axis=-1)    # W = N x C; v = C x C
            weights = self.dropout(weights)
            
            head    = tf.matmul(weights, v)
            heads.append(head)

        # Concatenate heads
        if len(heads) > 1:
            x = Concatenate(axis=-1)(heads)
        else:
            x = heads[0]

        # Combine heads
        x = self.proj(x)
        x = self.dropout(x)
        return x


class MLP(Layer):
    """MLP Layer: Dense -> Gelu -> Dense"""
    def __init__(self, dims, mlp_ratio=4, dropout_ratio=0.0, **kwargs):
        super().__init__(**kwargs)
        self.d1     = Dense(dims * mlp_ratio)
        self.bn1    = BatchNormalization(epsilon=1e-6)
        self.d2     = Dense(dims)
        self.bn2    = BatchNormalization(epsilon=1e-6)
        self.drop   = Dropout(dropout_ratio)

    def call(self, x):
        x = self.d1(x)
        x = self.bn1(x)
        x = gelu(x)
        x = self.drop(x)
        x = self.d2(x)
        x = self.bn2(x)
        x = self.drop(x)
        return x
    
class TransformerBlock(Layer):
    """
        x = x + MHA(x)
        x = x + MLP(x)
    """
    def __init__(self, dims, num_heads, mlp_ratio=4, dropout_ratio=0.0, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.ln_sc = LayerNormalization(epsilon=1e-6)
        self.mha = MultiHeadAttention(num_heads, dims, dropout_ratio)
        self.mlp = MLP(dims, mlp_ratio, dropout_ratio)
        
    def call(self, x, skip_connection=None):
        if skip_connection is None:
            # Self attention
            ln_x = self.ln1(x)
            x = x + self.mha(ln_x, ln_x)
        else:
            # Cross attention
            x = x + self.mha(self.ln1(x), self.ln_sc(skip_connection))
        x = x + self.mlp(self.ln2(x))
        return x
    
class AttentionModel:
    def __init__(self, 
                 dims,
                 depth,
                 heads,
                 num_patches=20,
                 num_classes=1,
                 num_channels=3,
                 num_vertices=153,
                 dropout=0.1,
                 branches=[],
                 activation='sigmoid'):
        self.inp = Input((num_patches, num_vertices, num_channels))
        self.depth = depth
        self.num_classes = num_classes
        self.dims = dims
        self.heads = heads
        self.num_patches = num_patches
        self.num_vertices = num_vertices
        self.activation = activation
        self.dropout = dropout
        self.branches=branches
        
    def reshape(self, x):
        B, P, V, C = get_shape(x)
        x = tf.reshape(x, (B, P, V * C))
        return x

    def __call__(self):
        x = self.inp
        
        x = GaussianNoise(1)(x)

        x = [x[..., y] for y in self.branches]
        
        # Patch Embeddings
        x = [PatchEmbeddings(self.num_patches, self.dims)(y) for y in x]
        
        # Independent analysis
        for i in range(self.depth[0]):
            x = [TransformerBlock(self.dims, self.heads, mlp_ratio=4., dropout_ratio=self.dropout)(y) for y in x]
            
        # Middle fusion
        if len(self.depth) > 1:
            for i in range(self.depth[1]):
                x = [TransformerBlock(self.dims, self.heads, mlp_ratio=4., dropout_ratio=self.dropout)(x[0], x[1]),
                     TransformerBlock(self.dims, self.heads, mlp_ratio=4., dropout_ratio=self.dropout)(x[1], x[0])]
        
        x = [Flatten()(y) for y in x]
        x = [LayerNormalization(epsilon=1e-6)(y) for y in x]

        if len(x) > 1: # Late fusion
            x = Concatenate(axis=-1)(x)
        else: # Early fusion
            x = x[0]

        x = Dense(self.dims)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = gelu(x)

        x = Dense(self.num_classes)(x)
        x = Activation(self.activation)(x)
        
        model = Model(inputs = self.inp, outputs = x)
        return model