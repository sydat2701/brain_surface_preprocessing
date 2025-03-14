import random
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import *
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_absolute_error, balanced_accuracy_score
'''
    REPRODUCIBILITY
'''

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

'''
    LR SCHEDULER
'''
def cosine_scheduler(initial_lr, min_lr, epochs_per_cycle):
    def scheduler(epoch, lr):
        return min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * (epoch % epochs_per_cycle) / epochs_per_cycle)) / 2
    return scheduler

'''
    CUSTOM CALLBACKS
'''

class ModelSavingCallback(Callback):
    def __init__(self, 
                 X_test, 
                 y_test, 
                 output_path,
                 verbose=False):
        super(ModelSavingCallback, self).__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.output_path = output_path
        self.verbose = verbose
        self.sen = 0.0
        self.spe = 0.0
        self.bacc = 0.0
        self.auc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_test, verbose=0)

        # Compute AUC before thresholding
        auc = roc_auc_score(self.y_test, y_pred)

        # Find best threshold based on bacc
        best_bacc = 0.0
        best_thres = 0.5
        for thres in [x * 0.05 for x in range(1, 20)]:
            test = (y_pred > thres)
            TN, FP, FN, TP = confusion_matrix(self.y_test, test).ravel()
            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)
            bacc = (sensitivity + specificity) / 2
            if bacc > best_bacc:
                best_thres = thres
                best_bacc = bacc
                
        y_pred = (y_pred > best_thres)

        # Compute metrics using confusion matrix
        TN, FP, FN, TP = confusion_matrix(self.y_test, y_pred).ravel()

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        bacc = (sensitivity + specificity) / 2

        if self.verbose:
            print(f"SEN: {sensitivity}, SPE: {specificity}, BACC: {bacc}, AUC: {auc}")
        # If validation bacc improved, update metrics and save model weights
        if bacc > self.bacc:
            if self.verbose:
                print(f"Validation F1 improved from {self.bacc} to {bacc}. Saving model weights to {self.output_path}")
            # Update best metrics
            self.sen = sensitivity
            self.spe = specificity
            self.bacc = bacc
            self.auc = auc
            # Save model weights
            self.model.save_weights(self.output_path)

'''
    DATA UTILITY
'''

def get_task_data(X, y, task):
    neg_idx = pos_idx = -1
    X_return = []
    y_return = []
    if task == 'cn_ad':
        neg_idx = 0
        pos_idx = 3
    elif task == 'cn_mci':
        neg_idx = 0
        pos_idx = (1, 2)

    for i, class_id in enumerate(y):
        if class_id == neg_idx:
            X_return.append(X[i])
            y_return.append(0)
        
        if isinstance(pos_idx, tuple):
            if class_id in pos_idx:
                X_return.append(X[i])
                y_return.append(1)
                
        elif class_id == pos_idx:
            X_return.append(X[i])
            y_return.append(1)
            
    return np.asarray(X_return), np.asarray(y_return)