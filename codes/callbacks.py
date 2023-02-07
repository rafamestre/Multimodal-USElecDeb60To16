# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:40:24 2022

@author: Himanshu Rawlani at https://github.com/himanshurawlani/hyper_fcn/blob/master/callbacks.py

@Adapted by: Rafael Mestre, r.mestre@soton.ac.uk

https://github.com/rafamestre/Multimodal-USElecDeb60To16
"""

import os
import tensorflow as tf
import logging
logger = logging.getLogger('hyper_pbt')

class TuneReporter(tf.keras.callbacks.Callback):
    """Tune Callback for Keras."""

    def __init__(self, reporter=None, freq="epoch", logs=None):
        """Initializer.
        Args:
            freq (str): Sets the frequency of reporting intermediate results.
                One of ["batch", "epoch"].
        """
        self.iteration = 0
        logs = logs or {}
        if freq not in ["batch", "epoch"]:
            raise ValueError("{} not supported as a frequency.".format(freq))
        self.freq = freq
        super(TuneReporter, self).__init__()

    def on_batch_end(self, batch, logs=None):
        """This will be called at the end of a batch, if that option is selected.
        The tune reporter is called and the metrics updated."""
        
        from ray import tune
        logs = logs or {}
        if not self.freq == "batch":
            return
        self.iteration += 1
        for metric in list(logs):
            if "loss" in metric and "neg_" not in metric:
                logs["neg_" + metric] = -logs[metric]
        if "acc" in logs:
            tune.report(keras_info=logs, mean_accuracy=logs["acc"])
        else:
            tune.report(keras_info=logs, mean_accuracy=logs.get("accuracy"))

    def on_epoch_end(self, batch, logs=None):
        """This will be called at the end of an epoch, if that option is selected.
        The tune reporter is called and the metrics updated."""

        from ray import tune
        logs = logs or {}
        if not self.freq == "epoch":
            return
        self.iteration += 1
        for metric in list(logs):
            if "loss" in metric and "neg_" not in metric:
                logs["neg_" + metric] = -logs[metric]
        if "acc" in logs:
            tune.report(keras_info=logs, 
                        val_loss=logs['val_loss'], 
                        acc=logs["acc"],
                        val_acc=logs["val_acc"],
                        auc=logs['auc'],
                        val_auc=logs['val_auc'])
        else:
            tune.report(keras_info=logs, 
                        val_loss=logs['val_loss'], 
                        acc=logs.get("accuracy"),
                        val_acc=logs.get("val_accuracy"),
                        auc=logs['auc'],
                        val_auc=logs['val_auc'])


def create_callbacks(final_run, model_path, monitor = 'loss', timestamp = None, 
                     model_name = None, run_nb = None):
    callbacks = []

    # If the loss function is monitored, the minimum value is taken
    # If accuracy or auc are monitored, the maximum value is taken
    if monitor == 'loss':
        mode = 'min'
    elif monitor in ['auc','acc']:
        mode = 'max'
    else:
        raise Exception('Monitor metric not valid. Must be "acc", "auc" or "loss".')
        
    # Creating early stopping callback
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor=f"val_{monitor}", 
                                                     patience=3,  
                                                     min_delta=1e-4, 
                                                     mode=mode, 
                                                     restore_best_weights=True, 
                                                     verbose=1)
    callbacks.append(earlystopping)

    if final_run:
        logger.info("Creating model checkpoint callback")
        # Make sure the 'results' directory exists
        os.makedirs(model_path, exist_ok=True)

        # Creating model checkpoint callback
        checkpoint_path = os.path.join(model_path,model_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        if not run_nb is None:
            checkpoint_path = os.path.join(checkpoint_path, timestamp+f'_model_run{run_nb}.h5')
        else:
            checkpoint_path = os.path.join(checkpoint_path, timestamp+'_model.h5')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                        monitor=f"val_{monitor}",
                                                        mode=mode,
                                                        save_best_only=True, 
                                                        verbose=1)
        callbacks.append(checkpoint)

    else:
        logger.info("Creating tune reporter callback")
        # Creating ray callback which reports metrics of the ongoing run
        # We choose to report metrics after epoch end using freq="epoch"
        # because val_loss is calculated just before the end of epoch
        tune_reporter = TuneReporter(freq="epoch")

        callbacks.append(tune_reporter)

    return callbacks
