'''
    IMPORT LIBRARIES
'''
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import StratifiedKFold
import gc
import random
import os
from utils import *

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.losses import BinaryFocalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow_addons.metrics import F1Score

from tensorflow.keras.callbacks import *
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_absolute_error, balanced_accuracy_score
import tensorflow_addons as tfa


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam
from model.attention_model import AttentionModel

'''
    CONFIGURATION
'''

EPOCH = 50
BATCH_SIZE = 8
SEED = 2022

seed_everything(SEED)
        
'''
    OPTIMIZATION
'''

X, y = np.load('./data_preprocessed/AD/data_roi_nopvc_ico2.npy').astype('float32'), np.load('./data_preprocessed/AD/labels.npy')
X = np.transpose(X, (0, 2, 3, 1)) # Batch - Channel - Patch - Vertices
print(f"Training data: {X.shape}" )

tasks = ['cn_ad', 'cn_mci']
for i, task in enumerate(tasks):
    X_ , y_, = get_task_data(X, y, task)
    kf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=2022)
    fold = 0
    BACC = []
    SEN  = []
    SPE  = []
    AUC_SCORE  = []
    for train_idx, test_idx in kf.split(X_, y_):
        K.clear_session()
        gc.collect()
        fold += 1
        print(f"***FOLD: {fold}***")

        X_train, y_train = X_[train_idx], np.expand_dims(y_[train_idx], axis=-1)
        X_test, y_test = X_[test_idx], np.expand_dims(y_[test_idx], axis=-1)
        
        seed_everything(2022)

        model = AttentionModel(dims=192,
                               depth=[3,3],
                               heads=3,
                               num_patches=X.shape[1],
                               num_classes=1,
                               num_channels=X.shape[-1],
                               num_vertices=X.shape[-2],
                               dropout=0.1,
                               branches=[slice(0, 3), slice(3, 5)],
                               activation='sigmoid')()

        callbacks = [ModelSavingCallback(X_test, y_test, f"model_checkpoints/model_{task}_{fold}_adni23.h5", verbose=False),
                     LearningRateScheduler(schedule=cosine_scheduler(1e-3, 1e-5, EPOCH), verbose=0)]

        model.compile(optimizer='adam', loss='binary_crossentropy')

        model.fit(X_train, 
                  y_train, 
                  batch_size=BATCH_SIZE, 
                  epochs=EPOCH, 
                  callbacks=callbacks, 
                  verbose=0)

        sensitivity       = callbacks[0].sen
        specificity       = callbacks[0].spe
        balanced_accuracy = callbacks[0].bacc
        auc               = callbacks[0].auc

        print(f"FOLD {fold}: BACC: {balanced_accuracy} SEN: {sensitivity} SPE: {specificity} AUC: {auc}")

        BACC.append(balanced_accuracy)
        SEN.append(sensitivity)
        SPE.append(specificity)
        AUC_SCORE.append(auc)
        
    print(f"Result for task {task}")
    print(f"OOF BACC: {BACC} {np.mean(BACC)} +- {np.std(BACC)}")
    print(f"OOF SEN: {SEN} {np.mean(SEN)} +- {np.std(SEN)}")
    print(f"OOF SPE: {SPE} {np.mean(SPE)} +- {np.std(SPE)}")
    print(f"OOF AUC: {AUC_SCORE} {np.mean(AUC_SCORE)} +- {np.std(AUC_SCORE)}")



