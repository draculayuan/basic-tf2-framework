import tensorflow as tf
from scipy.stats import spearmanr
import numpy as np

def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues + np.random.normal(0, 1e-7, col_pred.shape[0]), col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


class CustomCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, valid_data, batch_size=16, save_root=None):

        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]
        self.save_root = save_root
        self.batch_size = batch_size
        
    def on_train_begin(self, logs={}):
        self.valid_predictions = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.valid_predictions.append(
            self.model.predict(self.valid_inputs, batch_size=self.batch_size))
        
        rho_val = compute_spearmanr(
            self.valid_outputs, np.average(self.valid_predictions, axis=0))
        
        print("\nvalidation rho: %.4f" % rho_val)
        
        self.model.save_weights(self.save_root+f'bert-base-{epoch}.h5py')
        