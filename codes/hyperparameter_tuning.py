# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:36:51 2022

@author: Rafael Mestre, r.mestre@soton.ac.uk

https://github.com/rafamestre/Multimodal-USElecDeb60To16

"""


from pathlib import Path
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
import json
import traceback
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
#For ML models
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.preprocessing.sequence import pad_sequences


from hyperparameter_util_functions import obj_to_oh, preprocess_text_embeddings
from hyperparameter_util_functions import split_and_pad
from hyperparameter_util_functions import audio_padding
from hyperparameter_util_functions import text_model_BERT
from hyperparameter_util_functions import text_model_bilstm
from hyperparameter_util_functions import audio_model, audio_model_bilstm, audio_model_cnn
from hyperparameter_util_functions import multimodal_model, multimodal_model_cnn, multimodal_model_bilstm
from hyperparameter_util_functions import multimodal_model2
from hyperparameter_util_functions import prepare_argument_model
from hyperparameter_util_functions import NumpyEncoder
from callbacks import create_callbacks

from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray import tune
import logger as Logger
import argparse
import ray
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers import MedianStoppingRule
from ray.tune import CLIReporter

logger = Logger.get_logger('logger', './logs')


# Checking resources
physical_GPU = tf.config.list_physical_devices('GPU')
physical_CPU = tf.config.list_physical_devices('CPU')

if len(physical_GPU) > 0:
    HAS_GPU = True
else:
    HAS_GPU = False
    
if len(physical_CPU) > 0:
    HAS_CPU = True
else:
    HAS_CPU = False
    raise Exception('CPU not found.')

if HAS_GPU:
    main_device = '/gpu:0'
else:
    main_device = '/cpu:0'
    
    

# gpus = tf.config.list_physical_devices('GPU')
if HAS_GPU:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in physical_GPU:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(physical_GPU), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



# Global variables
NUM_SAMPLES = 1024 ## For the smoke test/finish fast

# These are all the models that can be run with hyperparameter tuning
ALLOWED_MODELS = ['bert',
                  'bilstm',
                  'audio',
                  'multimodal',
                  'audiocnn',
                  'audiobilstm',
                  'multimodalcnn',
                  'multimodalbilstm',
                  'multimodal2']

# These are the audio features that are considered for the audio models
ALLOWED_AUDIO_FEATURES = ['mfccs','centroids',
                          'rolloff','chroma',
                          'contrast','bandwidth']

# Maximum number of epochs to train the models
# Callbacks and early stopping are implemented, so it probably won't finish all the epochs
MAX_EPOCHS = 75




class MyTrainable:
    # Class object that is trainable with ray[tune]
    
    def __init__(self, results_dir, data_dir, model_type, finish_fast, monitor,
                 preprocess_path, encoder_path, timestamp, name, max_epochs = MAX_EPOCHS,
                 audio_dir = None, final_run=False, nb_outputs = 2, run_nb = None,
                 balanced = False, frac_data = 1, skip_audio_feature = None):
        
        # Initializing state variables for the run
        # "Finish fast" idea comes from 
        # https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/pbt_memnn_example.py

        self.data_dir = data_dir
        self.results_dir = results_dir
        self.final_run = final_run
        self.finish_fast = finish_fast
        self.model_type = model_type
        self.timestamp = timestamp
        self.name = name
        self.max_epochs = max_epochs
        self.run_nb = run_nb
        self.nb_outputs = nb_outputs
        self.filepath_audio = os.path.abspath(audio_dir)
        self.preprocess_path = preprocess_path
        self.encoder_path = encoder_path
        self.monitor = monitor
        self.balanced = balanced
        self.frac_data = frac_data
        self.glove_path = os.path.abspath(Path(r'./GloVe/')) # Cannot use relative path with workers
        self.skip_audio_feature = skip_audio_feature
        
        # if it's the final run, randomize the results each time to make statistics
        # if it's not, keep always the same seed for reproducibility and fair comparison,
        # so that each trial of the optimization uses the same training data
        if self.final_run:
            self.seed = np.random.randint(1e6)
        else:
            self.seed = 32
        
        # Since the text models do not require large amounts of data,
        # we read the data during initialisation, instead of in the function train()
        # This data is pickled and transferred to different workers.
        # This saves time because the data is loaded only once per optimisation.
        # The audio data is significantly larger and cannot be serialised easily
        # so we load it each time in the train() function
        if self.model_type == 'bert':
            self.text_train, self.text_test, self.y_train, self.y_test = self.get_data_BERT()
            self.other_config = {'preprocess_path': preprocess_path,
                                 'encoder_path': encoder_path,
                                 'nb_outputs': nb_outputs}

        elif self.model_type == 'bilstm':
            self.text_train, self.text_test, self.y_train, self.y_test = self.get_data_BiLSTM()
            self.other_config = {'embedding_matrix': self.embedding_matrix,
                                 'maxlen': self.maxlen,
                                 'nb_outputs': nb_outputs}



    def get_data_audio(self):
        # Gets the data for the audio and the multimodal models
        
        # TODO: make filename a parameter
        filename = 'allDebates_withAnnotations_all_timestamps.csv'
        
        # We get the dataset
        df = pd.read_csv(os.path.abspath(Path(self.data_dir,filename)))
        df_arguments = prepare_argument_model(df, tokenize = False)

        old_length = len(df_arguments)

        # We read the audio features here
        with open(Path(self.filepath_audio,'df_audio_features.pkl'), 'rb') as handle:
            df_features = pickle.load(handle)

        # We add the features to the dataset
        df_arguments = pd.merge(df_arguments,df_features,on=['ID'],how='outer')

        # Remove the Nones
        new_length_merging = len(df_arguments)
        df_arguments = df_arguments[~(df_arguments['audio_features'].isnull())]

        new_length = len(df_arguments)

        print(f'\n\nThe length of the dataframe went from {old_length} to {new_length_merging} after merging to {new_length} after removing nans.\n\n')

        # We get the maximum length of the features (audio length)
        shape = list()
        for i, row in df_arguments.iterrows():
            shape.append(row['audio_features'].shape[1])
                        

        # Take only the 99% percentile lenght
        self.max_shape_audio = int(np.percentile(shape, 99))
        
        if self.skip_audio_feature == 'mfccs':
            # mfccs have 23 coordinates and are the last features
            df_arguments['audio_features'] = df_arguments['audio_features'].apply(lambda x: x[:22])
        elif self.skip_audio_feature == 'chroma':
            # chroma ft have 12 coordinates and are from coordinate 10 to 21
            df_arguments['audio_features'] = df_arguments['audio_features'].apply(lambda x: 
                                                                                  np.concatenate((x[:10],x[22:])))
        elif self.skip_audio_feature == 'contrast':
            # spectral contrast have 7 coordinates and are from coordinate 3 to 9
            df_arguments['audio_features'] = df_arguments['audio_features'].apply(lambda x: 
                                                                                  np.concatenate((x[:3],x[9:])))
        elif self.skip_audio_feature == 'rolloff':
            # spectral contrast have 1 coordinates and is number 2
            df_arguments['audio_features'] = df_arguments['audio_features'].apply(lambda x: 
                                                                                  np.concatenate((x[:2],x[3:])))
        elif self.skip_audio_feature == 'bandwidth':
            # spectral bandwidth have 1 coordinates and is number 1
            df_arguments['audio_features'] = df_arguments['audio_features'].apply(lambda x: 
                                                                                  np.concatenate((x[:1],x[2:])))
        elif self.skip_audio_feature == 'centroids':
            # spectral centroids have 1 coordinates and is number 0
            df_arguments['audio_features'] = df_arguments['audio_features'].apply(lambda x: 
                                                                                  x[1:])

        if self.skip_audio_feature:
            logger.info(f'Skipping audio feature {self.skip_audio_feature}...')
            logger.info(f'New size of the audio features: {df_arguments.iloc[0].audio_features.shape[0]}.')
                
        # Features have been concatenated like this:
            # features = np.concatenate((spectral_centroids[-1], 
            #                            spectral_bandwidth[-1], 
            #                            spectral_rolloff[-1], 
            #                            spectral_contrast[-1], 
            #                            chroma_ft[-1], mfccs[-1]),axis=0)

        # We balance the dataset if needed
        # The seeds are a multiple of the main one for reproducibility
        if not self.finish_fast and self.balanced:
            logger.info('Balancing the dataset...')
            others = df_arguments[df_arguments['argument']=='Other']
            nb_other = len(others)
            arguments = df_arguments[df_arguments['argument']=='Argument']
            arguments = arguments.sample(n=nb_other, random_state=self.seed*2)
            df_arguments = pd.concat([others,arguments], axis=0)
            df_arguments = df_arguments.sample(frac=1, random_state=self.seed*3)
            logger.info(f'Now there are {len(df_arguments)} annotations, half of each class.')
        
        # If we want fractional datasets
        if not self.finish_fast and self.frac_data < 1:
            logger.info(f'Keeping only {self.frac_data*100}% of the dataset.')
            df_arguments = df_arguments.sample(frac=self.frac_data, random_state=self.seed*4)

        # We split the train and test data
        df_train, df_test = train_test_split(df_arguments, test_size=0.2,
                                             random_state = self.seed)

        y_test, text_test = df_test['argument'], df_test['Speech']
        y_train, text_train = df_train['argument'], df_train['Speech']
        
        # We padd the audio features
        padded_audio_train = np.array(audio_padding(df_train['audio_features'].values,self.max_shape_audio))
        padded_audio_test = np.array(audio_padding(df_test['audio_features'].values,self.max_shape_audio))

        # If this is a test and the finish fast variable is True, we reduce the number of samples
        if self.finish_fast:
            logger.info(f"Finishing fast. Reducing number of samples to {NUM_SAMPLES}")
            text_train, y_train = text_train[:NUM_SAMPLES], y_train[:NUM_SAMPLES]
            text_test, y_test = text_test[:NUM_SAMPLES], y_test[:NUM_SAMPLES]
            padded_audio_train, padded_audio_test = padded_audio_train[:NUM_SAMPLES,:,:], padded_audio_test[:NUM_SAMPLES,:,:]

        elif not self.finish_fast:
            logger.info("Not finishing fast. Full training performed.")


        # Labels are one-hot encoded now because pre-processing audio might have dropped some examples
        # if the audio could not be pre-processed for some reason (e.g. too short to extract features)
        # In that case, that whole example will be discarded and the lengths of the vector change
        split = y_train.shape[0]
        y = np.concatenate((y_train,y_test))
        y_oh = obj_to_oh(y)[0]
        y_train = y_oh[:split,:]
        y_test = y_oh[split:,:]


        return text_train, text_test, y_train, y_test, padded_audio_train, padded_audio_test


    def get_data_multimodal2(self):
        # Gets the data for the Bi-LSTM multimodal model
        
        # TODO: make filename a parameter
        filename = 'allDebates_withAnnotations_all_timestamps.csv'
        
        # We get the dataset
        df = pd.read_csv(os.path.abspath(Path(self.data_dir,filename)))
        df_arguments, nb_unique_words = prepare_argument_model(df, tokenize = True)
        
        old_length = len(df_arguments)
        
        # We read the audio features here
        with open(Path(self.filepath_audio,'df_audio_features.pkl'), 'rb') as handle:
            df_features = pickle.load(handle)

        # We add the features to the dataset
        df_arguments = pd.merge(df_arguments,df_features,on=['ID'],how='outer')

        # Remove the Nones
        new_length_merging = len(df_arguments)
        df_arguments = df_arguments[~(df_arguments['audio_features'].isnull())]

        new_length = len(df_arguments)

        print(f'\n\nThe length of the dataframe went from {old_length} to {new_length_merging} after merging to {new_length} after removing nans.\n\n')

        # We get the maximum length of the features (audio length)
        shape = list()
        for i, row in df_arguments.iterrows():
            shape.append(row['audio_features'].shape[1])
            # if features.shape[1] > max_shape:
            #     max_shape = features.shape[1]
        

        # Take only the 99% percentile lenght
        self.max_shape_audio = int(np.percentile(shape, 99))
        
        # Features have been concatenated like this:
            # features = np.concatenate((spectral_centroids[-1], 
            #                            spectral_bandwidth[-1], 
            #                            spectral_rolloff[-1], 
            #                            spectral_contrast[-1], 
            #                            chroma_ft[-1], mfccs[-1]),axis=0)

        # We balance the dataset if needed
        # The seeds are a multiple of the main one for reproducibility
        if not self.finish_fast and self.balanced:
            logger.info('Balancing the dataset...')
            others = df_arguments[df_arguments['argument']=='Other']
            nb_other = len(others)
            arguments = df_arguments[df_arguments['argument']=='Argument']
            arguments = arguments.sample(n=nb_other, random_state=self.seed*2)
            df_arguments = pd.concat([others,arguments], axis=0)
            df_arguments = df_arguments.sample(frac=1, random_state=self.seed*3)
            logger.info(f'Now there are {len(df_arguments)} annotations, half of each class.')
            

        # If we want fractional datasets
        if not self.finish_fast and self.frac_data < 1:
            logger.info(f'Keeping only {self.frac_data*100}% of the dataset.')
            df_arguments = df_arguments.sample(frac=self.frac_data, random_state=self.seed*4)


        

        # We train the tokeniser and create the embeddings only once
        X = list(df_arguments['tokenized'])
        tokenizer, self.embedding_matrix = preprocess_text_embeddings(X, nb_unique_words, 
                                                                  nb_dimensions=200, 
                                                                  embedding_type = 'Wikipedia',
                                                                  glove_path = self.glove_path)
        
        # Maximum length of a tokenised utterance to do paddings.
        # The great majority of sentences are of short length,
        # with only a couple of outliers that are very long
        # We use a 99% percentile (by default) to obtain the padding length
        self.maxlen = int(np.percentile([len(x) for x in X], 99))


        # Divide into train and test

        df_train, df_test = train_test_split(df_arguments, test_size=0.2,
                                             random_state = self.seed)

        y_test, text_test = df_test['argument'], df_test['tokenized']
        y_train, text_train = df_train['argument'], df_train['tokenized']
        
        padded_audio_train = np.array(audio_padding(df_train['audio_features'].values,self.max_shape_audio))
        padded_audio_test = np.array(audio_padding(df_test['audio_features'].values,self.max_shape_audio))


        # Tokenising
        text_train = tokenizer.texts_to_sequences(text_train)
        text_test = tokenizer.texts_to_sequences(text_test)

        text_train = pad_sequences(np.array(text_train), padding='post', maxlen=self.maxlen)
        text_test = pad_sequences(np.array(text_test), padding='post', maxlen=self.maxlen)


        # If this is a test and the finish fast variable is True, we reduce the number of samples
        if self.finish_fast:
            logger.info(f"Finishing fast. Reducing number of samples to {NUM_SAMPLES}")
            text_train, y_train = text_train[:NUM_SAMPLES], y_train[:NUM_SAMPLES]
            text_test, y_test = text_test[:NUM_SAMPLES], y_test[:NUM_SAMPLES]
            padded_audio_train, padded_audio_test = padded_audio_train[:NUM_SAMPLES,:,:], padded_audio_test[:NUM_SAMPLES,:,:]

        elif not self.finish_fast:
            logger.info("Not finishing fast. Full training performed.")


        # Labels are one-hot encoded now because pre-processing audio might have dropped some examples
        # if the audio could not be pre-processed for some reason (e.g. too short to extract features)
        # In that case, that whole example will be discarded and the lengths of the vector change
        split = y_train.shape[0]
        y = np.concatenate((y_train,y_test))
        y_oh = obj_to_oh(y)[0]
        y_train = y_oh[:split,:]
        y_test = y_oh[split:,:]


        return text_train, text_test, y_train, y_test, padded_audio_train, padded_audio_test



    def get_data_BiLSTM(self):
        # Gets the data for the Bi-LSTM model
        
        # TODO: make filename a parameter
        filename = 'allDebates_withAnnotations_all_timestamps.csv'
        
        # We get the dataset
        df = pd.read_csv(os.path.abspath(Path(self.data_dir,filename)))
        df_arguments, nb_unique_words = prepare_argument_model(df, tokenize = True)

        # We balance the dataset if needed
        # The seeds are a multiple of the main one for reproducibility
        if not self.finish_fast and self.balanced:
            logger.info('Balancing the dataset...')
            others = df_arguments[df_arguments['argument']=='Other']
            nb_other = len(others)
            arguments = df_arguments[df_arguments['argument']=='Argument']
            arguments = arguments.sample(n=nb_other, random_state=self.seed*2)
            df_arguments = pd.concat([others,arguments], axis=0)
            df_arguments = df_arguments.sample(frac=1, random_state=self.seed*3)
            logger.info(f'Now there are {len(df_arguments)} annotations, half of each class.')
        
        # If we want fractional datasets
        if not self.finish_fast and self.frac_data < 1:
            logger.info(f'Keeping only {self.frac_data*100}% of the dataset.')
            df_arguments = df_arguments.sample(frac=self.frac_data, random_state=self.seed*4)


        # We train the tokeniser and create the embeddings only once
        X = list(df_arguments['tokenized'])
        tokenizer, self.embedding_matrix = preprocess_text_embeddings(X, nb_unique_words, 
                                                                  nb_dimensions=200, 
                                                                  embedding_type = 'Wikipedia')
        
        # Maximum length of a tokenised utterance to do paddings.
        # The great majority of sentences are of short length,
        # with only a couple of outliers that are very long
        # We use a 99% percentile (by default) to obtain the padding length
        self.maxlen = int(np.percentile([len(x) for x in X], 99))
        # self.vocab_size = len(tokenizer.word_index) + 1 ##Number of words in the tokenizer
        
        X = np.array(df_arguments['tokenized'])
        y = df_arguments['argument'].values
        y, labels_dict_arg = obj_to_oh(y)
        y = np.array(y).astype(np.uint8)

        # Divide into train and test sets
        text_train, text_test, y_train, y_test = split_and_pad(X, y, 
                                                            tokenizer,
                                                            self.maxlen,
                                                            test_size=0.2,
                                                            random_state=self.seed)
        
        # If this is a test and the finish fast variable is True, we reduce the number of samples
        if self.finish_fast:
            logger.info(f"Finishing fast. Reducing number of samples to {NUM_SAMPLES}")
            text_train, y_train = text_train[:NUM_SAMPLES], y_train[:NUM_SAMPLES]
            text_test, y_test = text_test[:NUM_SAMPLES], y_test[:NUM_SAMPLES]

        elif not self.finish_fast:
            logger.info("Not finishing fast. Full training performed.")

        
        return text_train, text_test, y_train, y_test


    def get_data_BERT(self):
        # Gets the data for BERT model
        
        # TODO: make filename a parameter
        filename = 'allDebates_withAnnotations_all_timestamps.csv'
        
        # We get the dataset
        df = pd.read_csv(os.path.abspath(Path(self.data_dir,filename)))
        df_arguments = prepare_argument_model(df, tokenize = False)

        # We balance the dataset if needed
        # The seeds are a multiple of the main one for reproducibility
        if not self.finish_fast and self.balanced:
            logger.info('Balancing the dataset...')
            others = df_arguments[df_arguments['argument']=='Other']
            nb_other = len(others)
            arguments = df_arguments[df_arguments['argument']=='Argument']
            arguments = arguments.sample(n=nb_other, random_state=self.seed*2)
            df_arguments = pd.concat([others,arguments], axis=0)
            df_arguments = df_arguments.sample(frac=1, random_state=self.seed*3)
            logger.info(f'Now there are {len(df_arguments)} annotations, half of each class.')
        
        if not self.finish_fast and self.frac_data < 1:
            logger.info(f'Keeping only {self.frac_data*100}% of the dataset.')
            df_arguments = df_arguments.sample(frac=self.frac_data, random_state=self.seed*4)


        # We split the train and test data
        X = np.array(df_arguments['Speech'])
        y = df_arguments['argument'].values
        y, labels_dict_arg = obj_to_oh(y)
        y = np.array(y).astype(np.uint8)
        
        text_train, text_test, y_train, y_test = train_test_split(X, 
                                                                  y, 
                                                                  test_size=0.2, 
                                                                  random_state=self.seed)

        # If this is a test and the finish fast variable is True, we reduce the number of samples
        if self.finish_fast:
            logger.info(f"Finishing fast. Reducing number of samples to {NUM_SAMPLES}")

            text_train, y_train = text_train[:NUM_SAMPLES], y_train[:NUM_SAMPLES]
            text_test, y_test = text_test[:NUM_SAMPLES], y_test[:NUM_SAMPLES]

        elif not self.finish_fast:
            logger.info("Not finishing fast. Full training performed.")


        return text_train, text_test, y_train, y_test



    def train(self, config):
        # As of 10/12/2019: One caveat of using TF2.0 is that TF AutoGraph
        # functionality does not interact nicely with Ray actors. One way to get around
        # this is to `import tensorflow` inside the Tune Trainable, even if it's not used implicitly
        import tensorflow as tf
        import tensorflow_text as text
        
        # For the audio, it's better to load the data each time when training, otherwise
        # it cannot serialise the training because the data is too heavy, even if that means each worker spending
        # time loading it
        
        if 'audio' in self.model_type or self.model_type in ['multimodal','multimodalcnn','multimodalbilstm']:
            self.text_train, self.text_test, self.y_train, self.y_test,\
                self.padded_audio_train, self.padded_audio_test = self.get_data_audio()
                
            self.other_config = {'input_shape': self.padded_audio_train.shape[1:] + tuple([1]),
                                  'nb_outputs': self.nb_outputs}
            
            if 'multi' in self.model_type:
                self.other_config = {**self.other_config,
                                      **{'preprocess_path': self.preprocess_path,
                                        'encoder_path': self.encoder_path}}
        elif self.model_type == 'multimodal2':
            self.text_train, self.text_test, self.y_train, self.y_test,\
                self.padded_audio_train, self.padded_audio_test = self.get_data_multimodal2()
                
            self.other_config = {'input_shape': self.padded_audio_train.shape[1:] + tuple([1]),
                                  'nb_outputs': self.nb_outputs,
                                  'embedding_matrix': self.embedding_matrix,
                                  'maxlen': self.maxlen}




        
        print('Printing trial config file: \n\n')
        print(config)
        print('\n\n\n')

        
        # Create model
        if self.model_type=='bert':
            model = text_model_BERT({**config,**self.other_config})
        elif self.model_type == 'bilstm':
            model = text_model_bilstm({**config,**self.other_config})
        elif self.model_type == 'audio':
            model = audio_model({**config,**self.other_config})
        elif self.model_type == 'multimodal':
            model = multimodal_model({**config,**self.other_config})
        elif self.model_type == 'multimodalcnn':
            model = multimodal_model_cnn({**config,**self.other_config})
        elif self.model_type == 'multimodalbilstm':
            model = multimodal_model_bilstm({**config,**self.other_config})
        elif self.model_type == 'multimodal2':
            model = multimodal_model2({**config,**self.other_config})
        elif self.model_type == 'audiobilstm':
            model = audio_model_bilstm({**config,**self.other_config})
        elif self.model_type == 'audiocnn':
            model = audio_model_cnn({**config,**self.other_config})

        # print(model.summary())

        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']), 
                      loss = 'CategoricalCrossentropy', 
                      metrics=['acc',
                               tf.keras.metrics.AUC(multi_label=True,
                                                    num_labels=2,
                                                    name='auc')])
        
        # Create callbacks to be used during model training
        callbacks = create_callbacks(self.final_run, self.results_dir,
                                     self.monitor,
                                     self.timestamp, self.name,
                                     self.run_nb)

        
        
        # Start model training
        logger.info("Starting model training")
        if self.model_type in ['bert','bilstm']:
            input_data = self.text_train
        elif 'audio' in self.model_type:
            input_data = self.padded_audio_train
        elif 'multi' in self.model_type:
            input_data = [self.text_train,self.padded_audio_train]
        
        history = model.fit(input_data, self.y_train, 
                                batch_size=config['batch_size'], 
                                # epochs=config['epochs'], 
                                epochs=self.max_epochs, 
                                verbose=1, 
                                validation_split=0.2,
                                callbacks=callbacks)
        

        if self.final_run:
            self.plot_history(history)
            df_report = self.plot_metrics(model)
            return df_report

        # _, accuracy, auc = self.model.evaluate(text_test, y_test, verbose=0)

        # return history


    def plot_history(self, history):
        
        
        print('Plotting history.')
        # Convert the history.history dict to a pandas DataFrame:     
        hist_df = pd.DataFrame(history.history) 
        
        save_dir = os.path.join(self.results_dir,self.name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save to json:  
        hist_json_file = os.path.join(save_dir,f'{self.timestamp}_model_history_run{self.run_nb}.json') 
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)

        # And save to csv: 
        hist_csv_file = os.path.join(save_dir,f'{self.timestamp}_model_history_run{self.run_nb}.csv') 

        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
            
        # Plotting the loss vs val_loss
        fig = plt.figure(figsize=(10,6)) 
        ax=plt.gca()
        sns.set_style("whitegrid")
        sns.lineplot(data=hist_df[['loss','val_loss']], ax=ax)
        ax.legend(['Training', 'Validation'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        savename = os.path.join(save_dir,f'{self.timestamp}_model_loss_run{self.run_nb}') 
        fig.savefig(savename+".png", dpi=300)
        fig.savefig(savename+".svg", dpi=300)
        plt.close()


    def plot_metrics(self, model):
        
        print('Plotting metrics.')
        # This part figures out which time of model it is according to the
        # validation features introduced
        
        save_dir = os.path.join(self.results_dir,self.name)
        os.makedirs(save_dir, exist_ok=True)
        
        # We predict the labels of the validation set
        if self.model_type == 'bert' or self.model_type == 'bilstm':
            y_pred = model.predict(x=[self.text_test])

        elif 'audio' in self.model_type:
            y_pred = model.predict(x=[self.padded_audio_test])

        elif 'multi' in self.model_type:
            y_pred = model.predict(x=[self.text_test,self.padded_audio_test])
        
        # if file_appendix:
        #     savename += '_'+file_appendix
        
        # We transform the one-hot encoding of the form (x,x,x)
        # to a single label between of 0,1,2
        y_pred = np.argmax(y_pred,axis=1)
        y_true = np.argmax(self.y_test,axis=1)
        # We compute the confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        
        # This part of the code is aesthetics and is shared by every model
        group_counts = ["{0:0.0f}".format(value) for value in
                        cf_matrix.flatten()]
        
        group_ratios = [value*100 for value in
                        cf_matrix[0,:].flatten()/np.sum(cf_matrix[0,:])]
        
        group_ratios = group_ratios + [value*100 for value in
                        cf_matrix[1,:].flatten()/np.sum(cf_matrix[1,:])]
                
        group_percentages = ["{0:.2%}".format(value) for value in
                            cf_matrix[0,:].flatten()/np.sum(cf_matrix[0,:])]
        
        group_percentages = group_percentages + ["{0:.2%}".format(value) for value in
                            cf_matrix[1,:].flatten()/np.sum(cf_matrix[1,:])]
                
        labels = [f"{v1}\n{v2}" for v1, v2 in
                  zip(group_counts,group_percentages)]
        
        group_ratios = np.asarray(group_ratios).reshape(2,2)
        labels = np.asarray(labels).reshape(2,2)
        fig = plt.figure(figsize=(6,6)) 
        ax = plt.gca()
        sns.heatmap(group_ratios, annot=labels, fmt="", cmap='Blues', ax=ax, annot_kws={"size":15}, vmin=0, vmax=100)
        
        ax.set_xlabel('Predicted labels', fontsize = 14, labelpad= 10)
        ax.set_ylabel('True labels', fontsize = 14, labelpad = 10)
        ax.set_title('Confusion Matrix', fontsize = 14)
        ax.xaxis.set_ticklabels(['argument','other'], fontsize = 14)
        ax.yaxis.set_ticklabels(['argument','other'], fontsize = 14)
        ax.collections[0].colorbar.set_label("Percentage (%)", fontsize=15)
        cbar_ax = ax.figure.axes[-1].yaxis
        cbar_ax.set_ticklabels(cbar_ax.get_ticklabels(), fontsize=15)
        
        # We save the confusion matrix of the results
        savename = os.path.join(save_dir,
                                f'{self.timestamp}_model_')

        fig.savefig(savename+f"cm_run{self.run_nb}.png", dpi=300)
        fig.savefig(savename+f"cm_run{self.run_nb}.svg", dpi=300)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=['argument','other'],output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        # There is a column named "support", which is the number of occurences in each class
        # Since this is confusing with our "support" label, we change the name
        df_report = df_report.rename(columns = {'support':'occurences'})
        
        # We save the classification report in .csv and we print it on screen
        df_report.to_csv(savename+f"metrics_run{self.run_nb}.csv")
        
        print(df_report)
        
        return df_report






def parse_args(args):
    """
    Example command:
    $ python train.py --train-dir dataset/train --val-dir dataset/val --optimize True --samples 100
    """
    
    parser = argparse.ArgumentParser(description='Hyperparameter optimisation')

    parser.add_argument('--results-dir', type=str, 
                        help='Path to results directory.',
                        default=Path('results'))
    parser.add_argument('--data-dir', type=str, 
                        help='Path to data directory.',
                        default=Path('Multimodal ElecDeb60To16'))
    parser.add_argument('--audio-dir', type=str, 
                        help='Path to the directory where the audio segments are.',
                        default=Path(r'./Original'))
    parser.add_argument('--config-path', type=str, 
                        help='Model config path (considered only when optimize=False).', 
                        default='./default_config.json')
    parser.add_argument('--optimize', type=str, 
                        help='Flag to run hyperparameter search.', 
                        default="True")
    parser.add_argument('--samples', type=int, 
                        help='Number of times to sample from the hyperparameter space.', 
                        default=2)
    
    parser.add_argument('--frac-data', type=float, 
                        help='Fraction of the dataset to keep. Must be a float between 0 and 1.', 
                        default=1)
    
    parser.add_argument('--monitor', type=str, 
                        help='Metric to monitor. Must be "loss", "auc" or "acc". Defaults to "loss".', 
                        default='loss')
    
    parser.add_argument('--best-metric', type=str, 
                        help='Metric to select the best model. Should be "loss" (not recommended if different architectures are considered), "auc" or "acc". Defaults to "acc".', 
                        default='acc')
    

    parser.add_argument('--num-gpus', type=float, 
                        help='Number of GPUs to use.', 
                        default=0)
    parser.add_argument('--num-cpus', type=float, 
                        help='Number of GPUs to use.', 
                        default=1)

    parser.add_argument('--local-dir', type=str, 
                        help='Local directory to save the logs.', 
                        default='./ray_results')
    parser.add_argument('--config-file', type=str, 
                        help='Config file to be used for the run. Only works if optimize is set to False.', 
                        default=None)
    
    parser.add_argument("--scheduler",  type=str,
                        help="Scheduler to use. Defaults to None. Allowed: ASHA, median, None.",
                        default=None)



    parser.add_argument("--smoke-test", 
                        action="store_true", 
                        help="Finish quickly for testing")

    parser.add_argument("--balanced", 
                        action="store_true", 
                        help="Balance the dataset to the smallest class.")

    parser.add_argument("--local-mode", 
                        action="store_true", 
                        help="Local model with only one thread for faster execution during testing.")
    
    parser.add_argument("--model",  type=str,
                        help="Name of the model to do hyperparametner training on.",
                        default = 'audio')
    
    parser.add_argument("--name",  type=str,
                        help="Name of the experiment. If empty, will default to 'model'.",
                        default = None)

    parser.add_argument("--max-epochs",  type=int,
                        help="Maximum number of epochs to train for. Defaults to 50.",
                        default = MAX_EPOCHS)

    parser.add_argument("--nb-runs",  type=int,
                        help="Number of runs to do for the final training. Defaults to 5.",
                        default = 5)
    
    parser.add_argument("--skip-audio-feature",  type=str,
                        help="Name of the audio feature to skip for ablation study. Defaults to None.",
                        default = None)


    return parser.parse_args(args)



def create_search_space(model_type):
    # Creating hyperopt search space
    # Possible parameters: https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions
    
    # First the general hyperparameters
    search_space = {
               "lr": hp.loguniform("lr", np.log(0.000001), np.log(0.01)),
               	"batch_size": hp.choice("batch_size", [16,32,64]),
               # "epochs": hp.choice('epochs', np.arange(1, MAX_EPOCHS, dtype=int)),
    }

    #Then the parameters that depend on the model type
    if model_type=='bert':
        search_space = {**search_space,
                         **{"nb_neurons_bert_dense": hp.choice('nb_neurons_bert_dense', 
                                                          [16,32,64,128,256]),
                            "hidden_activation": hp.choice('hidden_activation', 
                                                           ['relu','sigmoid','tanh']),
                            "dropout_text": hp.choice("dropout_text", 
                                                      [0,0.1,0.2,0.3,0.4,0.5,
                                                       0.6,0.7,0.8,0.9]),
                            "trainable": hp.choice("trainable", [True,False]),
                             }
                         }
    elif model_type=='bilstm':
        search_space = {**search_space,
                         **{
                            "hidden_activation": hp.choice('hidden_activation', 
                                                           ['relu','sigmoid','tanh']),
                            "dropout_text": hp.choice("dropout_text", 
                                                      [0,0.1,0.2,0.3,0.4,0.5,
                                                       0.6,0.7,0.8,0.9]),
                            "trainable": hp.choice("trainable", [True,False]),
                            "nb_neurons_bisltm_layer": hp.choice('nb_neurons_bisltm_layer', 
                                                                 [16,32,64,128,256]),
                            "nb_neurons_bisltm_dense": hp.choice('nb_neurons_bisltm_dense', 
                                                                 [16,32,64,128,256]),
                             }
                         }

    elif model_type=='audio' or model_type=='multimodal':
        search_space = {**search_space,
                         **{
                            "hidden_activation": hp.choice('hidden_activation', 
                                                           ['relu','sigmoid','tanh']),
                            "dropout_audio": hp.choice("dropout_audio", 
                                                      [0,0.1,0.2,0.3,0.4,0.5,
                                                       0.6,0.7,0.8,0.9]),
                            "nb_neurons_bisltm_layer": hp.choice('nb_neurons_bisltm_layer', 
                                                                 [16,32,64,128,256]),
                            "nb_neurons_dense": hp.choice('nb_neurons_dense', 
                                                          [16,32,64,128,256]),
                            "conv1_filters": hp.choice('conv1_filters', 
                                                       [4,8,16,32,64]),
                            "conv2_filters": hp.choice('conv2_filters', 
                                                       [4,8,16,32,64]),
                            "conv1_kernel": hp.choice('conv1_kernel', 
                                                      [1,3,5,7]),
                            "conv2_kernel": hp.choice('conv2_kernel', 
                                                      [1,3,5,7]),
                            # "strides": hp.choice('strides', 
                            #                      [1,2,3]),
                            "pool1_size": hp.choice('pool1_size', 
                                                      [2,4]),
                            "pool2_size": hp.choice('pool2_size', 
                                                      [2,4]),
                             }
                         }

    elif model_type=='audiocnn' or model_type=='multimodalcnn':
        search_space = {**search_space,
                         **{
                            "hidden_activation": hp.choice('hidden_activation', 
                                                           ['relu','sigmoid','tanh']),
                            "dropout_audio": hp.choice("dropout_audio", 
                                                      [0,0.1,0.2,0.3,0.4,0.5,
                                                       0.6,0.7,0.8,0.9]),
                            "nb_neurons_dense": hp.choice('nb_neurons_dense', 
                                                          [16,32,64,128,256]),
                            "conv1_filters": hp.choice('conv1_filters', 
                                                       [4,8,16,32,64]),
                            "conv2_filters": hp.choice('conv2_filters', 
                                                       [4,8,16,32,64]),
                            "conv1_kernel": hp.choice('conv1_kernel', 
                                                      [1,3,5,7]),
                            "conv2_kernel": hp.choice('conv2_kernel', 
                                                      [1,3,5,7]),
                            # "strides": hp.choice('strides', 
                            #                      [1,2,3]),
                            "pool1_size": hp.choice('pool1_size', 
                                                      [2,4]),
                            "pool2_size": hp.choice('pool2_size', 
                                                      [2,4]),
                             }
                         }

    elif model_type=='audiobilstm' or model_type=='multimodalbilstm':
        search_space = {**search_space,
                         **{
                            "hidden_activation": hp.choice('hidden_activation', 
                                                           ['relu','sigmoid','tanh']),
                            "dropout_audio": hp.choice("dropout_audio", 
                                                      [0,0.1,0.2,0.3,0.4,0.5,
                                                       0.6,0.7,0.8,0.9]),
                            "nb_neurons_dense": hp.choice('nb_neurons_dense', 
                                                          [16,32,64,128,256]),
                            "nb_neurons_bisltm_layer": hp.choice('nb_neurons_bisltm_layer', 
                                                                 [16,32,64,128,256]),
                             }
                         }
        
    elif model_type=='multimodal2':
        search_space = {**search_space,
                         **{
                            "hidden_activation": hp.choice('hidden_activation', 
                                                           ['relu','sigmoid','tanh']),
                            "dropout_audio": hp.choice("dropout_audio", 
                                                      [0,0.1,0.2,0.3,0.4,0.5,
                                                       0.6,0.7,0.8,0.9]),
                            "nb_neurons_bisltm_text": hp.choice('nb_neurons_bisltm_text', 
                                                                 [16,32,64,128,256]),
                            "nb_neurons_bisltm_audio": hp.choice('nb_neurons_bisltm_audio', 
                                                                 [16,32,64,128,256]),
                            "nb_neurons_dense": hp.choice('nb_neurons_dense', 
                                                          [16,32,64,128,256]),
                            "conv1_filters": hp.choice('conv1_filters', 
                                                       [4,8,16,32,64]),
                            "conv2_filters": hp.choice('conv2_filters', 
                                                       [4,8,16,32,64]),
                            "conv1_kernel": hp.choice('conv1_kernel', 
                                                      [1,3,5,7]),
                            "conv2_kernel": hp.choice('conv2_kernel', 
                                                      [1,3,5,7]),
                            # "strides": hp.choice('strides', 
                            #                      [1,2,3]),
                            "pool1_size": hp.choice('pool1_size', 
                                                      [2,4]),
                            "pool2_size": hp.choice('pool2_size', 
                                                      [2,4]),
                             }
                         }


    # If it's multimodal add some config on top of the audio ones
    if 'multi' in model_type:
        search_space = {**search_space,
                         **{
                            "dropout_text": hp.choice("dropout_text", 
                                                      [0,0.1,0.2,0.3,0.4,0.5,
                                                       0.6,0.7,0.8,0.9]),
                            "dropout_final": hp.choice("dropout_final", 
                                                      [0,0.1,0.2,0.3,0.4,0.5,
                                                       0.6,0.7,0.8,0.9]),
                            "trainable": hp.choice("trainable", [True,False]),

                             }
                         }


    return search_space

#TODO: number of trials? check https://datascience.stackexchange.com/questions/87905/is-there-a-rule-of-thumb-for-a-sufficient-number-of-trials-for-hyperparameter-se


def main(args=None):
    
    # Parse command line arguments.
    if args is None:
        args = sys.argv[1:]
        
    args = parse_args(args)
    
    if not args.model.lower() in ALLOWED_MODELS:
        raise Exception('Introduce a valid model type.')
    
    if args.skip_audio_feature:
        if not args.skip_audio_feature in ALLOWED_AUDIO_FEATURES:
            raise Exception('Introduce a valid audio feature.')
    
    args.model = args.model.lower()

    if not args.name:
        name = args.model
    else:
        name = args.name

    if args.nb_runs < 1:
        raise Exception('The number of runs needs to be at least 1.')

    # Get a date-time stamp to save the results
    dateTimeObj = datetime.now()
    timestamp = str(dateTimeObj.year) + '-' + str(dateTimeObj.month) + \
        '-'+ str(dateTimeObj.day)+'_'+str(dateTimeObj.hour) + '-' + \
            str(dateTimeObj.minute)+ '-' + str(dateTimeObj.second)
            

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    other_config = {#Only for the BERT model
                    'preprocess_path': os.path.abspath('./models/bert_en_uncased_preprocess_3/'),
                    'encoder_path' : os.path.abspath('./models/bert_en_uncased_L-12_H-768_A-12_4')
                    }

    if args.optimize == "True":
        # If we do hyperparameter tuning
        logger.info("Initializing ray")
        try:
            ray.init(configure_logging=False)
        except:
            ray.shutdown()
            ray.init(configure_logging=False)


        logger.info("Initializing ray search space")
        search_space = create_search_space(args.model)
        
        
        logger.info("Initializing scheduler and search algorithms")
        if args.monitor == 'loss':
            mode = 'min'
        elif args.monitor in ['acc','auc']:
            mode = 'max'
        else:
            raise Exception('Monitor metric should be "loss", "auc" or "acc".')
            
        
        if args.scheduler == 'asha':
            logger.info("Running with ASHA scheduler.")
            scheduler = AsyncHyperBandScheduler(time_attr='training_iteration',
                                                metric=f"val_{args.monitor}",
                                                mode=mode,
                                                grace_period=5)
        elif args.scheduler == 'median':
            logger.info("Running with Median Stop scheduler.")
            scheduler = MedianStoppingRule(time_attr='training_iteration',
                                           metric=f"val_{args.monitor}",
                                           mode=mode,
                                           grace_period = 5)
        elif not args.scheduler:
            logger.info("Running without scheduler (only early stopping).")
            scheduler = None

        # Use bayesian optimisation provided by hyperopt
        hyperopt_search = HyperOptSearch(search_space,
                                         metric=f"val_{args.monitor}",
                                         mode=mode)
                                    # points_to_evaluate=default_config)

        
        # 'training_iteration' is incremented every time `train` is called
        # because of the keras callback that calls tune.report
        stopping_criteria = {"training_iteration": 1 if args.smoke_test else args.max_epochs}


        # Initialize Trainable for hyperparameter tuning
        logger.info("Initializing ray Trainable")

        trainer = MyTrainable(os.path.abspath(args.results_dir), 
                            os.path.abspath(args.data_dir), 
                            # os.path.abspath(args.snapshot_dir), 
                            args.model,
                            args.smoke_test, 
                            args.monitor,
                            other_config['preprocess_path'],
                            other_config['encoder_path'],
                            timestamp,
                            name,
                            max_epochs = args.max_epochs,
                            audio_dir = args.audio_dir,
                            final_run=False,
                            balanced=args.balanced,
                            frac_data=args.frac_data,
                            skip_audio_feature=args.skip_audio_feature)

        # Reporter to show on command line/output window
        reporter = CLIReporter(
            metric_columns=["loss", "accuracy", "training_iteration", "auc"])

        # Establish the resources per trial
        resources_per_trial = {'cpu': args.num_cpus,
                               'gpu': args.num_gpus}
        
        local_dir = os.path.abspath(args.local_dir)
        


        logger.info("Starting hyperparameter tuning")
        analysis = tune.run(trainer.train, 
                            name = name,
                            search_alg = hyperopt_search,
                            verbose=1, 
                            # stop= CustomStopper(args.smoke_test),
                            num_samples=args.samples,
                            scheduler=scheduler,
                            raise_on_failed_trial=False,
                            resources_per_trial=resources_per_trial,
                            # config=config,
                            local_dir=local_dir,
                            progress_reporter = reporter,
                            stop=stopping_criteria,
                            # checkpoint_freq=1,
                            # checkpoint_at_end=True

                            )

        if args.best_metric == 'loss':
            mode_best = 'min'
        elif args.best_metric in ['acc','auc']:
            mode_best = 'max'
        else:
            raise Exception('best_metric should be "loss", "auc" or "acc".')
            
        best_config = analysis.get_best_config(metric=f"val_{args.best_metric}", mode=mode_best)
        logger.info(f'Best config: {best_config}')

        if best_config is None:
            logger.error('Optimization failed')
        else:
            logger.info("Saving best model config")
            
            save_dir = os.path.join(args.results_dir,name)
            os.makedirs(save_dir, exist_ok=True)

            with open(os.path.join(save_dir, timestamp+f'_best_config_{args.best_metric}.json'), 'w') as f:
                json.dump(best_config, f, indent=4, cls=NumpyEncoder)
            
            # Save also the other metric just in case
            if args.best_metric == 'acc':
                args.best_metric = 'auc'
            elif args.best_metric == 'auc':
                args.best_metric == 'acc'
            
            if args.best_metric != 'loss':
                best_config_alternative = analysis.get_best_config(metric=f"val_{args.best_metric}", mode=mode_best)
                with open(os.path.join(save_dir, timestamp+f'_best_config_{args.best_metric}.json'), 'w') as f:
                    json.dump(best_config_alternative, f, indent=4, cls=NumpyEncoder)
            
            
            logger.info("Saving analysis dataframe")
            # Get a dataframe for the last reported results of all of the trials
            global df
            df = analysis.results_df
            print(df.head())

            # save_dir = os.path.join(local_dir,name)
            df.to_csv(os.path.join(save_dir,timestamp+'_'+args.model+'_results_all.csv'))
            
            # Get a dataframe for the max accuracy seen for each trial
            # df_acc = analysis.dataframe(metric="val_auc", mode="max")
            # print(df_acc.head())
            # df.to_csv(os.path.join(args.results_dir,datetimenow+'_results_acc.csv'))

            # Get a dict mapping {trial logdir -> dataframes} for all trials in the experiment.
            # all_dataframes = analysis.trial_dataframes
            # print(all_dataframes)
            # Get a list of trials
            # trials = analysis.trials
            # print(trials)

            # analysis = ExperimentAnalysis(local_dir)

            
            
            logger.info(f"Refitting the model on best configuration {args.nb_runs} time(s).")
            df_report = []
            
            for i in range(1,args.nb_runs+1):
                logger.info("Waiting for GPU/CPU memory cleanup")
                time.sleep(10)

                logger.info(f'Doing run number {i}.')
                trainer = MyTrainable(os.path.abspath(args.results_dir), 
                                    os.path.abspath(args.data_dir), 
                                    # os.path.abspath(args.snapshot_dir), 
                                    args.model,
                                    args.smoke_test, 
                                    args.monitor,
                                    other_config['preprocess_path'],
                                    other_config['encoder_path'],
                                    timestamp,
                                    name,
                                    max_epochs = args.max_epochs,
                                    audio_dir = args.audio_dir,
                                    final_run=True,
                                    run_nb = i,
                                    balanced=args.balanced,
                                    frac_data=args.frac_data,
                                    skip_audio_feature=args.skip_audio_feature)


                df_report.append(trainer.train(best_config))
                
            df_report_all = pd.concat(df_report)
            #Get the average, std and sem of all the reports
            df_report_stats = df_report_all.groupby(df_report_all.index).agg(['mean','std','sem'])
            df_report_stats.columns = ['_'.join(col) for col in df_report_stats.columns]
            
            
            df_report_stats.to_csv(os.path.join(save_dir,
                                               f'{timestamp}_model_metrics_av.csv'))
            
            
    else:
        # If this is a normal training in which we only use the hyperparameters already optimised
        
        if not args.config_file:
            raise Exception('Optimize was False but there was no configuration file.')
        
        try:
            with open(args.config_file, 'r') as f:
                default_config = json.load(f)
        except Exception as e:
            print(e)
            raise Exception('Configuration file could not be opened.')

        logger.info(f"Training the model on default config {args.nb_runs} time(s).")
        df_report = []
                
        for i in range(1,args.nb_runs+1):
            logger.info("Waiting for GPU/CPU memory cleanup")
            time.sleep(3)

            logger.info(f'Doing run number {i}.')

            trainer = MyTrainable(os.path.abspath(args.results_dir), 
                                    os.path.abspath(args.data_dir), 
                                    # os.path.abspath(args.snapshot_dir), 
                                    args.model,
                                    args.smoke_test, 
                                    args.monitor,
                                    other_config['preprocess_path'],
                                    other_config['encoder_path'],
                                    timestamp,
                                    name,
                                    max_epochs = args.max_epochs,
                                    audio_dir = args.audio_dir,
                                    final_run=True,
                                    run_nb = i,
                                    balanced = args.balanced,
                                    frac_data=args.frac_data,
                                    skip_audio_feature=args.skip_audio_feature)

            
            df_report.append(trainer.train(default_config))
        
        rows = df_report[0].index
        df_report_all = pd.concat(df_report)
        #Get the average, std and sem of all the reports
        df_report_stats = df_report_all.groupby(df_report_all.index).agg(['mean','std','sem'])
        df_report_stats.columns = ['_'.join(col) for col in df_report_stats.columns]
        #The rows are disordered, this fixes it
        df_report_stats = df_report_stats.loc[rows]
        
        save_dir = os.path.join(args.results_dir,name)
        # Save config file
        with open(os.path.join(save_dir, timestamp+'_config_repeat.json'), 'w') as f:
            json.dump(default_config, f, indent=4, cls=NumpyEncoder)

        
        df_report_stats.to_csv(os.path.join(save_dir,
                                           f'{timestamp}_model_metrics_av.csv'))
            


    logger.info("Training completed")

if __name__ == "__main__":
    try:
        main()
    except:
        logger.error(traceback.format_exc())
        raise










