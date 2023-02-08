# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:52:54 2022

@author: Rafael Mestre, r.mestre@soton.ac.uk

https://github.com/rafamestre/Multimodal-USElecDeb60To16
"""





from pathlib import Path
import pandas as pd
# import sys
import os
import numpy as np
import json

#For ML models
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional
from tensorflow.keras.layers import Embedding, Dense
# from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow_hub as hub
# import tensorflow_text #for BERT uncased to work
from tensorflow.keras.preprocessing.text import Tokenizer


import nltk
from nltk.corpus import stopwords
from django.contrib.admin.utils import flatten
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder




def tokenize_nltk(text):
    '''
    Function to tokenize a string of text using the NLTK tokenizer

    Parameters
    ----------
    text : str
        Text to tokenize.

    Returns
    -------
    tokenized : str
        Tokenized text.

    '''
    text = text.lower()
    tokenized = nltk.word_tokenize(text)
    
    return tokenized


def clean_stopwords(text):
    '''
    Cleans a tokenized sentence removing stopwords. he list elements containing
        a stopword are revoved.

    Parameters
    ----------
    text : list of str
        Tokenized sentence in the form of a list of strings.

    Returns
    -------
    clean : list of str
        Cleaned sentence by removing stopwords.

    '''
    stop_words = set(stopwords.words('english'))
    clean = [w for w in text if not w in stop_words]
    
    return clean


def tokenise_utterances(text):
    """
    Tokenize and clean the text input.

    Parameters
    ----------
    text : list or pandas series
        Text to be processed, can be a list of strings or a pandas series of strings.

    Returns
    -------
    test_data : pandas dataframe
        Dataframe with columns 'text', 'tokenized', and 'clean' where each row represents the original text, its 
        tokenized form, and its cleaned form (removed stopwords) respectively.
    len(unique_words) : int
        Total number of unique words in the cleaned text.

    """
    
    print('Tokenising utterances...\n')
    test_data = pd.DataFrame({'text':text})
    test_data['tokenized'] = test_data.apply(lambda x: tokenize_nltk(x['text']), axis=1)
    test_data['clean'] = test_data.apply(lambda x: clean_stopwords(x['tokenized']), axis=1)
    print('Finished.')
    
    all_words = flatten([[w for w in sent] for sent in test_data['clean'].values])
    unique_words = set(all_words)
    print('There are a total of {} unique words.\n'.format(len(unique_words)))

    return test_data, len(unique_words)



def obj_to_oh(labels):
    """
    One-hot encoding function to encode the relation labels.

    Parameters
    ----------
    labels : numpy.ndarray
        Array of labels to one-hot encode of shape (N,).

    Returns
    -------
    labels_oh : numpy.ndarray
        Array of one-hot encoded labels of shape of shape (N,M),
        where M is the number of unique labels found.

    """
    labels = np.array(labels.tolist())    
    unique_labels = np.unique(labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    oe = OneHotEncoder(sparse=False)
    labels = labels.reshape(len(labels), 1)
    labels_oh = oe.fit_transform(labels)
    unique_labels_oh = oe.fit_transform(unique_labels.reshape(len(unique_labels), 1))
    labels_dict = dict(zip(unique_labels, unique_labels_oh))
    print('There are a total of {} labels with the following names:'.format(len(unique_labels)))
    [print('\t'+u) for u in labels_dict.keys()]
    return labels_oh, labels_dict

def preprocess_text_embeddings(X, nb_unique_words, nb_dimensions=200, 
                               embedding_type = 'Wikipedia', 
                               embedding_path = Path(r'./GloVe/')):
    """
    Preprocess the input text data by converting them into embeddings.
    
    Parameters
    ----------
    X : list of strings
        List of text data to preprocess.
    nb_unique_words : int
        Number of unique words to include in the text preprocessing.
    nb_dimensions : int, optional
        Number of dimensions for the embeddings (default is 200).
    embedding_type : str, optional
        Type of embeddings to use (default is 'Wikipedia').
    embedding_path : pathlib path, optional
        Path to the embeddings (default is Path(r'./GloVe/')).
    
    Returns
    -------
    tokenizer : keras_preprocessing.text.Tokenizer
        Tokenizer object that was fit on the input `X`.
    embedding_matrix : numpy.ndarray
        Embedding matrix of shape (nb_unique_words + 1, nb_dimensions)
    """

    
    print('Fitting word tokeniser... \n')
    tokenizer = Tokenizer(num_words=nb_unique_words+1)
    tokenizer.fit_on_texts(X)
    print('Finished.\n')

    vocab_size = len(tokenizer.word_index) + 1 # Number of words in the tokenizer

    print('Creating {} word embeddings...\n'.format(embedding_type))
    embedding_matrix, embeddings_dictionary, noncovered_elements = create_embeddings(tokenizer,
                                                                                     nb_dimensions,
                                                                                     embedding_type,
                                                                                     embedding_path=embedding_path)
    print('Finished.\n')
    print('Testing...')
    for word in ['clinton', 'sad', ',']:
        try:
            print('{}: {}'.format(word, tokenizer.word_index[word]))
            
        except:
            print(word+' not found.')
    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    print('{}% of elements covered by the pre-trained model. \n'.format(round(100*nonzero_elements / vocab_size, 4)))


    return tokenizer, embedding_matrix


def create_embeddings(tokenizer,nb_dimensions, source = 'Twitter', 
                      embedding_path = Path(r'./GloVe/')):
    
    '''
    Creates the embedding matrix from a GloVe file
    
    Parameters
    ----------
    tokenizer : keras_preprocessing.text.Tokenizer
        Tokenizer object that was fit on the input data.
    nb_dimensions : int
        Number of dimensions for the embeddings.
    source : str, optional
        Type of embeddings to use (default is 'Twitter').
    embedding_path : pathlib path, optional
        Path to the embeddings (default is Path(r'./GloVe/')).
        
    Returns
    -------
    embedding_matrix : numpy.ndarray
        Embedding matrix of shape (vocab_size, nb_dimensions).
    embeddings_dictionary : dict
        Dictionary that maps word to embedding vector.
    noncovered_elements : list
        List of words that are not included in the embeddings_dictionary.
    '''

    global embeddings_dictionary
    noncovered_elements = list()
        
    if source == 'Twitter':
        filename = Path('glove.twitter/glove.twitter.27B.'+str(nb_dimensions)+'d.txt')
    elif source == 'Wikipedia':
        filename = Path('glove.6B/glove.6B.'+str(nb_dimensions)+'d.txt')
    else:
        raise ValueError('The source is not acceptable. Should be either Twitter or Wikipedia.')
    
    source_file = Path(embedding_path,filename)
    if not os.path.exists(source_file):
        if not os.path.exists(source_file):
            print(source_file)
            raise Exception('Source file cannot be found. Perhaps the number of dimensions is wrong.')
        
    with open(source_file, encoding="utf8") as glove_file:
        embeddings_dictionary = dict()
        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = np.asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions
    
    # Creates embedding matrix for each word that's in the test data
    # and in the training corpus
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, nb_dimensions))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
        else:
            noncovered_elements.append(word)
            embedding_matrix[index] = np.random.rand(nb_dimensions)
    return embedding_matrix,embeddings_dictionary,noncovered_elements



def split_and_pad(X,y,tokenizer,maxlen,test_size = 0.2, random_state= 32):
    """
    Splits and pads the input data into training and testing datasets.
    
    Parameters
    ----------
    X : array-like
        An array-like object containing the input data to be split and padded.
    y : array-like
        An array-like object containing the target data to be split and padded.
    tokenizer : keras_preprocessing.text.Tokenizer
        A tokenizer object that is used to tokenize the input data.
    maxlen : int
        The maximum length of the padded sequences.
    test_size : float, optional (default=0.2)
        The fraction of the data to be used for testing.
    random_state : int, optional (default=32)
        The seed for the random number generator.
        
    Returns
    -------
    X_train : array
        An array of padded and tokenized training data.
    X_test : array
        An array of padded and tokenized testing data.
    y_train : array
        An array of training target data.
    y_test : array
        An array of testing target data.
    """

    #Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    #Tokenising
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    #Padding
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    return X_train, X_test, y_train, y_test



def prepare_argument_model(df, tokenize = False):
    """
    Prepare the dataframe for argument model.

    Parameters
    ----------
    df (pd.DataFrame): The original dataframe.
    tokenize (bool, optional): If True, tokenize the text. Default is False.

    Returns
    -------
    df_arguments : pd.DataFrame
        The dataframe with cleaned text and labels.
    nb_unique_words (int, optional)
        The number of unique words in the tokenized text. 
        Returned only if tokenize is True.

    """
    
    
    #Copy the dataframe and tokenise the text
    df_arguments = df.copy()
    
    #Eliminate the tags, uh's and remove rows that are empty
    df_arguments = clean_text_tags(df_arguments, column = 'Speech') 
    df_arguments = df_arguments[~(df_arguments['Component'].isnull())]
    
    
    #Remove hyphens and periods
    df_arguments['Speech'] = df_arguments['Speech'].replace('[-.‒–—…]', '', regex=True)
    
    #Label premise and claims as arguments, rest as other
    df_arguments['argument'] = df_arguments['Component'].apply(lambda x: 'Other' if x == 'O' else 'Argument')

    if tokenize:
        tokenised_utterances, nb_unique_words = tokenise_utterances(df_arguments['Speech'])
        df_arguments = pd.concat([df_arguments,tokenised_utterances], axis=1)
        return df_arguments, nb_unique_words
    
    else:
        return df_arguments




def clean_text_tags(df, column = 'Speech'):
    """
    Cleans the text from tags between brackets and the "uh's" in speech synthesis.

    Parameters
    ----------
    df (pd.DataFrame): A dataframe containing the speech.
    column (str, optional): The name of the column in `df` containing the speech. 
                            Default is 'Speech'.

    Returns
    -------
    df (pd.DataFrame): A cleaned dataframe containing the speech, without the tags 
                       between brackets and the "uh's".
    """

    #Cleans the text from tags between brackets for the speech synthesis
    #and the "uh's" from some speeches
    
    #This eliminates the parts between brackets and the "uh's"
    df[column] = df[column].replace('[(\[].*?[)\]]|([^A-aZ-z][uU][Hh])', '', regex=True)
    
    #This eliminates the rows which are now empty, maybe because they only
    #contained a tag like [APPLAUSE]
    df = df[df[column].apply(lambda x: any(c.isalpha() for c in x))]
    
    return df



def insert_timestamps(df, filepath_videos = Path(r'..\videos'), 
                      video_list_filename = 'YoutubeLinks.csv',
                      filepath_alignment = Path(r'..\alignment')):
    """
    Inserts the timestamps into the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with the debates.
    filepath_videos : Path, optional
        The filepath where the videos are stored, by default Path(r'..\videos').
    video_list_filename : str, optional
        The filename of the csv file containing the video links, by default 'YoutubeLinks.csv'.
    filepath_alignment : Path, optional
        The filepath where the alignments are stored, by default Path(r'..\alignment').
    
    Returns
    -------
    df : pd.DataFrame
        A dataframe containing the speech and the timestamps.
    """


    df_links = pd.read_csv(Path(filepath_videos,video_list_filename))

    df_list = []
    
    for d in df['Document'].unique():
        
        try:
            df_this = df[df['Document']==d]
            f_align = d + '_' + df_links[df_links['Debate']==d]['Name'].values[0] + '_syncmap.csv'
            df_timestamps = pd.read_csv(Path(filepath_alignment, f_align), header=None, 
                                        names=['id','begin','end','text'])
        except:
            continue
    
        df_timestamps_no_silence = df_timestamps[df_timestamps['id'].str.contains('f')]
        if len(df_timestamps_no_silence) != len(df_this):
            raise Exception('The timestamps and the dataframe have different number of sentences for {}'.format(d))
        else:
            print('Transcript for debate {} has the correct number of sentences.'.format(d))
    
        df_this = df_this.assign(Timestamp_ID= df_timestamps_no_silence['id'].values )
        df_this = df_this.assign(Begin_s= df_timestamps_no_silence['begin'].values )
        df_this = df_this.assign(End_s= df_timestamps_no_silence['end'].values )
    
        df_list.append(df_this)

        df_all = pd.concat(df_list)

    return df_all





def audio_padding(audio_features,max_shape):
    """
    Padding function for the audio features, since they can have different lengths.

    Parameters
    ----------
    audio_features : list
        List of numpy.ndarray with the audio features of each sentence
        of shape (45, T), where T is variable depending on the duration of the 
        utterance. If T < max_shape, trailing 0's are added to the vector.
    max_shape : FLOAT
        Shape of the longest sentence to compare and add 0's.

    Returns
    -------
    padded_audio_features : list
        List of numpy.ndarray with the same audio features padded with
        trailing 0's. Each element of the list now has shape (45,max_shape).

    """
    padded_audio_features = []
    for features in audio_features:
        if features.shape[1] <= max_shape:
            features = np.concatenate((features,np.zeros((features.shape[0],(max_shape - features.shape[1])))), axis = 1)
        elif features.shape[1] > max_shape:
            features = features[:,:max_shape]
        features = features.T
        padded_audio_features.append(features)
        #print (features.shape)
    return padded_audio_features


def text_model_bilstm(config):

    """
    Create a Bi-LSTM model for text classification.
    
    Parameters
    ----------
    config (dict): A dictionary containing model hyperparameters.
    
    Returns
    -------
    model : tf.keras.Model
        Text-only Bi-LSTM model.
    """

    nb_neurons_bisltm_layer = config['nb_neurons_bisltm_layer']
    nb_neurons_bisltm_dense = config['nb_neurons_bisltm_dense']
    trainable = config['trainable']
    hidden_activation = config['hidden_activation']
    embedding_matrix = config['embedding_matrix']
    maxlen = config['maxlen']
    nb_outputs = config['nb_outputs']
    dropout_text = config['dropout_text']
    
    vocab_size, nb_dimensions = embedding_matrix.shape
    
    deep_inputs = Input(shape=(maxlen,))
    embedding_layer = Embedding(vocab_size, nb_dimensions, weights=[embedding_matrix], 
                                input_length = maxlen, trainable=trainable)(deep_inputs)
    
    BiLSTM_Layer = Bidirectional(LSTM(nb_neurons_bisltm_layer))(embedding_layer)
    
    X_text = tf.keras.layers.Dropout(rate=dropout_text)(BiLSTM_Layer)

    dense_layer_1 = Dense(nb_neurons_bisltm_dense, activation=hidden_activation)(X_text) 
    dense_layer_2 = Dense(nb_outputs, activation='softmax')(dense_layer_1) #Softmax is better for mutually exclusive classes
    model = Model(inputs=deep_inputs, outputs=dense_layer_2)
    
    
    return model



def text_model_BERT(config):
    """
    Text-only model.

    Parameters
    ----------
    config (dict): A dictionary containing model hyperparameters.

    Returns
    -------
    model : tf.keras.Model
        Text-only model.

    """
    
    dropout_text = config.get('dropout_text')
    trainable = config.get('trainable')
    dense_units = int(config.get('nb_neurons_bert_dense'))
    hidden_activation = config.get('hidden_activation')
    nb_outputs = config['nb_outputs']

    #Inputs
    In_text = tf.keras.layers.Input(shape = (), dtype=tf.string, name='text')

    #text part: BERT model + dense layer for reshaping output (dense layer is shared between the two sentences)
    # bert_preprocess_model = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
    bert_preprocess_model = hub.KerasLayer(config['preprocess_path'])
    
    preproc_text = bert_preprocess_model(In_text)
    encoder = hub.KerasLayer(config['encoder_path'], 
                              trainable=trainable, name='BERT_encoder')
    
    # encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3', 
    #                           trainable=False, name='BERT_encoder')

    enc_outputs = encoder(preproc_text)
    X_text = enc_outputs['pooled_output']
    X_text = tf.keras.layers.Dropout(rate=dropout_text)(X_text)

    X_fin = tf.keras.layers.Dense(units = dense_units)(X_text)
    
    X_fin = tf.keras.layers.Activation(hidden_activation)(X_fin)

    out = tf.keras.layers.Dense(units = nb_outputs, activation = tf.nn.softmax)(X_fin)
    model = tf.keras.Model(inputs=In_text, outputs=out)
    return model





def audio_model(config):
    """
    Audio-only model. For details see the main paper.

    Parameters
    ----------
    config (dict): A dictionary containing model hyperparameters.

    Returns
    -------
    model : tf.keras.Model
        Audio-only model (full).

    """
    input_shape = config['input_shape']
    dropout_audio = config['dropout_audio']
    nb_neurons_bisltm_layer = config['nb_neurons_bisltm_layer']
    nb_outputs = config['nb_outputs']
    strides = 1
    hidden_activation = config['hidden_activation']
    nb_neurons_dense = config['nb_neurons_dense']
    conv1_filters = config['conv1_filters']
    conv2_filters = config['conv2_filters']
    conv1_kernel = config['conv1_kernel']
    conv2_kernel = config['conv2_kernel']
    pool1_size = config['pool1_size']
    pool2_size = config['pool2_size']


    #Inputs
    In_audio = tf.keras.layers.Input(shape = input_shape)

    #audio part
    Bi_LSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nb_neurons_bisltm_layer,
                                                                 return_sequences=True,
                                                                 kernel_regularizer=tf.keras.regularizers.L2(l2=0.5)))
    #Att = tf.keras.layers.Attention()
    f0 = tf.keras.layers.Flatten()
    dr0 = tf.keras.layers.Dropout(rate=dropout_audio)

    X_seq = Bi_LSTM(In_audio[:,:,:,0])
    #X_seq = Att([X_seq,X_seq])
    X_seq = f0(X_seq)
    X_seq = dr0(X_seq)

    #cnn part for audio. 
    
    conv1 = tf.keras.layers.Conv2D(filters = conv1_filters, kernel_size = conv1_kernel, 
                                   strides = strides, padding = 'valid')
    bn1 = tf.keras.layers.BatchNormalization()
    a1 = tf.keras.layers.Activation(hidden_activation)
    dr1 = tf.keras.layers.Dropout(rate=dropout_audio)
    mp1 = tf.keras.layers.MaxPool2D(pool_size=pool1_size,strides=strides,
                                    padding='valid')
    conv2 = tf.keras.layers.Conv2D(filters = conv2_filters, kernel_size = conv2_kernel, 
                                   strides = strides, padding = 'valid')
    bn2 = tf.keras.layers.BatchNormalization()
    a2 = tf.keras.layers.Activation(hidden_activation)
    dr2 = tf.keras.layers.Dropout(rate=dropout_audio)
    mp2 = tf.keras.layers.MaxPool2D(pool_size=pool2_size,strides=strides,
                                    padding='valid')
    f = tf.keras.layers.Flatten()

    X = conv1(In_audio)
    X = bn1(X)
    X = a1(X)
    X = dr1(X)
    X = mp1(X)
    X = conv2(X)
    X = bn2(X)
    X = a2(X)
    X = dr2(X)
    X = mp2(X)
    X = f(X)

    X_fin = tf.keras.layers.Concatenate()([X_seq,X])
    X_fin = tf.keras.layers.Dense(units = nb_neurons_dense)(X_fin)
    X_fin = tf.keras.layers.BatchNormalization()(X_fin)
    X_fin = tf.keras.layers.Activation(hidden_activation)(X_fin)
    X_fin = tf.keras.layers.Dropout(rate=dropout_audio)(X_fin)

    out = tf.keras.layers.Dense(units = nb_outputs, activation = tf.nn.softmax)(X_fin)
    model = tf.keras.Model(inputs=[In_audio], outputs=out)
    return model


def audio_model_cnn(config):
    """
    Audio-only model with only cnns. Not analysed in the paper.

    Parameters
    ----------
    config (dict): A dictionary containing model hyperparameters.

    Returns
    -------
    model : tf.keras.Model
        Audio-only model with only cnn.

    """
    input_shape = config['input_shape']
    dropout_audio = config['dropout_audio']
    # nb_neurons_bisltm_layer = config['nb_neurons_bisltm_layer']
    nb_outputs = config['nb_outputs']
    strides = 1
    hidden_activation = config['hidden_activation']
    nb_neurons_dense = config['nb_neurons_dense']
    conv1_filters = config['conv1_filters']
    conv2_filters = config['conv2_filters']
    conv1_kernel = config['conv1_kernel']
    conv2_kernel = config['conv2_kernel']
    pool1_size = config['pool1_size']
    pool2_size = config['pool2_size']

    #Inputs
    In_audio = tf.keras.layers.Input(shape = input_shape)

    
    conv1 = tf.keras.layers.Conv2D(filters = conv1_filters, kernel_size = conv1_kernel, 
                                   strides = strides, padding = 'valid')
    bn1 = tf.keras.layers.BatchNormalization()
    a1 = tf.keras.layers.Activation(hidden_activation)
    dr1 = tf.keras.layers.Dropout(rate=dropout_audio)
    mp1 = tf.keras.layers.MaxPool2D(pool_size=pool1_size,strides=strides,
                                    padding='valid')
    conv2 = tf.keras.layers.Conv2D(filters = conv2_filters, kernel_size = conv2_kernel, 
                                   strides = strides, padding = 'valid')
    bn2 = tf.keras.layers.BatchNormalization()
    a2 = tf.keras.layers.Activation(hidden_activation)
    dr2 = tf.keras.layers.Dropout(rate=dropout_audio)
    mp2 = tf.keras.layers.MaxPool2D(pool_size=pool2_size,strides=strides,
                                    padding='valid')
    f = tf.keras.layers.Flatten()

    X = conv1(In_audio)
    X = bn1(X)
    X = a1(X)
    X = dr1(X)
    X = mp1(X)
    X = conv2(X)
    X = bn2(X)
    X = a2(X)
    X = dr2(X)
    X = mp2(X)
    X = f(X)

    # X_fin = tf.keras.layers.Concatenate()([X_seq,X])
    X_fin = tf.keras.layers.Dense(units = nb_neurons_dense)(X)
    X_fin = tf.keras.layers.BatchNormalization()(X_fin)
    X_fin = tf.keras.layers.Activation(hidden_activation)(X_fin)
    X_fin = tf.keras.layers.Dropout(rate=dropout_audio)(X_fin)

    out = tf.keras.layers.Dense(units = nb_outputs, activation = tf.nn.softmax)(X_fin)
    model = tf.keras.Model(inputs=[In_audio], outputs=out)
    return model


def audio_model_bilstm(config):
    """
    Audio-only model with only bi-lstm. Not analysed in the paper.

    Parameters
    ----------
    config (dict): A dictionary containing model hyperparameters.

    Returns
    -------
    model : tf.keras.Model
        Audio-only model with only bi-lstm.

    """
    input_shape = config['input_shape']
    dropout_audio = config['dropout_audio']
    nb_neurons_bisltm_layer = config['nb_neurons_bisltm_layer']
    nb_outputs = config['nb_outputs']
    # strides = 1
    hidden_activation = config['hidden_activation']
    nb_neurons_dense = config['nb_neurons_dense']
    # conv1_filters = config['conv1_filters']
    # conv2_filters = config['conv2_filters']
    # conv1_kernel = config['conv1_kernel']
    # conv2_kernel = config['conv2_kernel']
    # pool1_size = config['pool1_size']
    # pool2_size = config['pool2_size']

    #Inputs
    In_audio = tf.keras.layers.Input(shape = input_shape)

    #audio part
    Bi_LSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nb_neurons_bisltm_layer,
                                                                  return_sequences=True,
                                                                  kernel_regularizer=tf.keras.regularizers.L2(l2=0.5)))
    #Att = tf.keras.layers.Attention()
    f0 = tf.keras.layers.Flatten()
    dr0 = tf.keras.layers.Dropout(rate=dropout_audio)

    X_seq = Bi_LSTM(In_audio[:,:,:,0])
    #X_seq = Att([X_seq,X_seq])
    X_seq = f0(X_seq)
    X_seq = dr0(X_seq)


    # X_fin = tf.keras.layers.Concatenate()([X_seq,X])
    X_fin = tf.keras.layers.Dense(units = nb_neurons_dense)(X_seq)
    X_fin = tf.keras.layers.BatchNormalization()(X_fin)
    X_fin = tf.keras.layers.Activation(hidden_activation)(X_fin)
    X_fin = tf.keras.layers.Dropout(rate=dropout_audio)(X_fin)

    out = tf.keras.layers.Dense(units = nb_outputs, activation = tf.nn.softmax)(X_fin)
    model = tf.keras.Model(inputs=[In_audio], outputs=out)
    return model



def multimodal_model(config):
    """
    Multimoal audio-text model based on BERT. For details see the main paper.

    Parameters
    ----------
    config (dict): A dictionary containing model hyperparameters.

    Returns
    -------
    model : tf.keras.Model
        Multimodal audio-text model based on BERT.
    """
    
    input_shape = config['input_shape']
    dropout_audio = config['dropout_audio']
    nb_neurons_bisltm_layer = config['nb_neurons_bisltm_layer']
    nb_outputs = config['nb_outputs']
    strides = 1
    hidden_activation = config['hidden_activation']
    nb_neurons_dense = config['nb_neurons_dense']
    conv1_filters = config['conv1_filters']
    conv2_filters = config['conv2_filters']
    conv1_kernel = config['conv1_kernel']
    conv2_kernel = config['conv2_kernel']
    pool1_size = config['pool1_size']
    pool2_size = config['pool2_size']
    preprocess_path = config['preprocess_path']
    encoder_path = config['encoder_path']
    trainable = config['trainable']
    dropout_text = config['dropout_text']
    dropout_final = config['dropout_final']
    
    #Inputs
    In_text = tf.keras.layers.Input(shape = (), dtype=tf.string, name='text')
    
    In_audio = tf.keras.layers.Input(shape = input_shape, name='audio')


    # text part: BERT model + dense layer for reshaping output (dense layer is shared between the two sentences)
    # bert_preprocess_model = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
    bert_preprocess_model = hub.KerasLayer(preprocess_path)
    
    preproc_text = bert_preprocess_model(In_text)
    encoder = hub.KerasLayer(encoder_path, 
                              trainable=trainable, name='BERT_encoder')

    enc_outputs = encoder(preproc_text)
    X_text = enc_outputs['pooled_output']
    X_text = tf.keras.layers.Dropout(rate=dropout_text)(X_text)




    #audio part
    Bi_LSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nb_neurons_bisltm_layer,
                                                                 return_sequences=True,
                                                                 kernel_regularizer=tf.keras.regularizers.L2(l2=0.5)))
    #Att = tf.keras.layers.Attention()
    f0 = tf.keras.layers.Flatten()
    dr0 = tf.keras.layers.Dropout(rate=dropout_audio)

    X_seq = Bi_LSTM(In_audio[:,:,:,0])
    #X_seq = Att([X_seq,X_seq])
    X_seq = f0(X_seq)
    X_seq = dr0(X_seq)

    #cnn part for audio. 
    
    conv1 = tf.keras.layers.Conv2D(filters = conv1_filters, kernel_size = conv1_kernel, 
                                   strides = strides, padding = 'valid')
    bn1 = tf.keras.layers.BatchNormalization()
    a1 = tf.keras.layers.Activation(hidden_activation)
    dr1 = tf.keras.layers.Dropout(rate=dropout_audio)
    mp1 = tf.keras.layers.MaxPool2D(pool_size=pool1_size,strides=strides,
                                    padding='valid')
    conv2 = tf.keras.layers.Conv2D(filters = conv2_filters, kernel_size = conv2_kernel, 
                                   strides = strides, padding = 'valid')
    bn2 = tf.keras.layers.BatchNormalization()
    a2 = tf.keras.layers.Activation(hidden_activation)
    dr2 = tf.keras.layers.Dropout(rate=dropout_audio)
    mp2 = tf.keras.layers.MaxPool2D(pool_size=pool2_size,strides=strides,
                                    padding='valid')
    f = tf.keras.layers.Flatten()

    X = conv1(In_audio)
    X = bn1(X)
    X = a1(X)
    X = dr1(X)
    X = mp1(X)
    X = conv2(X)
    X = bn2(X)
    X = a2(X)
    X = dr2(X)
    X = mp2(X)
    X = f(X)

    X_fin = tf.keras.layers.Concatenate()([X_text,X_seq,X])
    X_fin = tf.keras.layers.Dense(units = nb_neurons_dense)(X_fin)
    X_fin = tf.keras.layers.BatchNormalization()(X_fin)
    X_fin = tf.keras.layers.Activation(hidden_activation)(X_fin)
    X_fin = tf.keras.layers.Dropout(rate=dropout_final)(X_fin)



    out = tf.keras.layers.Dense(units = nb_outputs, activation = tf.nn.softmax)(X_fin)
    model = tf.keras.Model(inputs=[In_text,In_audio], outputs=out)
    return model

def multimodal_model_cnn(config):
    """
    Multimoal audio-text model based on BERT with only CNN in audio pipeline.
    Not analysed in paper.

    Parameters
    ----------
    config (dict): A dictionary containing model hyperparameters.

    Returns
    -------
    model : tf.keras.Model
        Multimoal audio-text model based on BERT with only CNN in audio pipeline.

    """

    
    input_shape = config['input_shape']
    dropout_audio = config['dropout_audio']
    # nb_neurons_bisltm_layer = config['nb_neurons_bisltm_layer']
    nb_outputs = config['nb_outputs']
    strides = 1
    hidden_activation = config['hidden_activation']
    nb_neurons_dense = config['nb_neurons_dense']
    conv1_filters = config['conv1_filters']
    conv2_filters = config['conv2_filters']
    conv1_kernel = config['conv1_kernel']
    conv2_kernel = config['conv2_kernel']
    pool1_size = config['pool1_size']
    pool2_size = config['pool2_size']
    preprocess_path = config['preprocess_path']
    encoder_path = config['encoder_path']
    trainable = config['trainable']
    dropout_text = config['dropout_text']
    dropout_final = config['dropout_final']
    
    #Inputs
    In_text = tf.keras.layers.Input(shape = (), dtype=tf.string, name='text')
    
    In_audio = tf.keras.layers.Input(shape = input_shape, name='audio')


    #text part: BERT model + dense layer for reshaping output (dense layer is shared between the two sentences)
    # bert_preprocess_model = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
    bert_preprocess_model = hub.KerasLayer(preprocess_path)
    
    preproc_text = bert_preprocess_model(In_text)
    encoder = hub.KerasLayer(encoder_path, 
                              trainable=trainable, name='BERT_encoder')

    enc_outputs = encoder(preproc_text)
    X_text = enc_outputs['pooled_output']
    X_text = tf.keras.layers.Dropout(rate=dropout_text)(X_text)




    #audio part

    #cnn part for audio. 
    
    conv1 = tf.keras.layers.Conv2D(filters = conv1_filters, kernel_size = conv1_kernel, 
                                   strides = strides, padding = 'valid')
    bn1 = tf.keras.layers.BatchNormalization()
    a1 = tf.keras.layers.Activation(hidden_activation)
    dr1 = tf.keras.layers.Dropout(rate=dropout_audio)
    mp1 = tf.keras.layers.MaxPool2D(pool_size=pool1_size,strides=strides,
                                    padding='valid')
    conv2 = tf.keras.layers.Conv2D(filters = conv2_filters, kernel_size = conv2_kernel, 
                                   strides = strides, padding = 'valid')
    bn2 = tf.keras.layers.BatchNormalization()
    a2 = tf.keras.layers.Activation(hidden_activation)
    dr2 = tf.keras.layers.Dropout(rate=dropout_audio)
    mp2 = tf.keras.layers.MaxPool2D(pool_size=pool2_size,strides=strides,
                                    padding='valid')
    f = tf.keras.layers.Flatten()

    X = conv1(In_audio)
    X = bn1(X)
    X = a1(X)
    X = dr1(X)
    X = mp1(X)
    X = conv2(X)
    X = bn2(X)
    X = a2(X)
    X = dr2(X)
    X = mp2(X)
    X = f(X)

    X_fin = tf.keras.layers.Concatenate()([X_text,X])
    X_fin = tf.keras.layers.Dense(units = nb_neurons_dense)(X)
    X_fin = tf.keras.layers.BatchNormalization()(X_fin)
    X_fin = tf.keras.layers.Activation(hidden_activation)(X_fin)
    X_fin = tf.keras.layers.Dropout(rate=dropout_final)(X_fin)



    out = tf.keras.layers.Dense(units = nb_outputs, activation = tf.nn.softmax)(X_fin)
    model = tf.keras.Model(inputs=[In_text,In_audio], outputs=out)
    return model


def multimodal_model_bilstm(config):
    
    """
    Multimoal audio-text model based on BERT with only bi-lstm in audio pipeline.
    Not analysed in paper.

    Parameters
    ----------
    config (dict): A dictionary containing model hyperparameters.

    Returns
    -------
    model : tf.keras.Model
        Multimoal audio-text model based on BERT with only bi-lstm in audio pipeline.

    """
    
    input_shape = config['input_shape']
    dropout_audio = config['dropout_audio']
    nb_neurons_bisltm_layer = config['nb_neurons_bisltm_layer']
    nb_outputs = config['nb_outputs']
    hidden_activation = config['hidden_activation']
    nb_neurons_dense = config['nb_neurons_dense']
    preprocess_path = config['preprocess_path']
    encoder_path = config['encoder_path']
    trainable = config['trainable']
    dropout_text = config['dropout_text']
    dropout_final = config['dropout_final']
    
    #Inputs
    In_text = tf.keras.layers.Input(shape = (), dtype=tf.string, name='text')
    
    In_audio = tf.keras.layers.Input(shape = input_shape, name='audio')


    #text part: BERT model + dense layer for reshaping output (dense layer is shared between the two sentences)
    # bert_preprocess_model = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
    bert_preprocess_model = hub.KerasLayer(preprocess_path)
    
    preproc_text = bert_preprocess_model(In_text)
    encoder = hub.KerasLayer(encoder_path, 
                              trainable=trainable, name='BERT_encoder')

    enc_outputs = encoder(preproc_text)
    X_text = enc_outputs['pooled_output']
    X_text = tf.keras.layers.Dropout(rate=dropout_text)(X_text)




    #audio part
    Bi_LSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nb_neurons_bisltm_layer,
                                                                  return_sequences=True,
                                                                  kernel_regularizer=tf.keras.regularizers.L2(l2=0.5)))
    #Att = tf.keras.layers.Attention()
    f0 = tf.keras.layers.Flatten()
    dr0 = tf.keras.layers.Dropout(rate=dropout_audio)

    X_seq = Bi_LSTM(In_audio[:,:,:,0])
    #X_seq = Att([X_seq,X_seq])
    X_seq = f0(X_seq)
    X_seq = dr0(X_seq)


    X_fin = tf.keras.layers.Concatenate()([X_text,X_seq])
    X_fin = tf.keras.layers.Dense(units = nb_neurons_dense)(X_seq)
    X_fin = tf.keras.layers.BatchNormalization()(X_fin)
    X_fin = tf.keras.layers.Activation(hidden_activation)(X_fin)
    X_fin = tf.keras.layers.Dropout(rate=dropout_final)(X_fin)



    out = tf.keras.layers.Dense(units = nb_outputs, activation = tf.nn.softmax)(X_fin)
    model = tf.keras.Model(inputs=[In_text,In_audio], outputs=out)
    return model


def multimodal_model2(config):
    """
    Multimodal audio-text model with bi-lstm text module and full audio module.
    For details see the main paper.

    Parameters
    ----------
    config (dict): A dictionary containing model hyperparameters.

    Returns
    -------
    model : tf.keras.Model
        Multimodal audio-text model with bi-lstm text module and full audio module.

    """

    
    hidden_activation = config['hidden_activation']
    dropout_audio = config['dropout_audio']
    nb_neurons_bisltm_text = config['nb_neurons_bisltm_text']
    nb_neurons_bisltm_audio = config['nb_neurons_bisltm_audio']
    nb_neurons_dense = config['nb_neurons_dense']
    conv1_filters = config['conv1_filters']
    conv2_filters = config['conv2_filters']
    conv1_kernel = config['conv1_kernel']
    conv2_kernel = config['conv2_kernel']
    pool1_size = config['pool1_size']
    pool2_size = config['pool2_size']
    dropout_text = config['dropout_text']
    dropout_final = config['dropout_final']
    trainable = config['trainable']

    # Other config
    input_shape = config['input_shape']
    nb_outputs = config['nb_outputs']
    strides = 1
    
    embedding_matrix = config['embedding_matrix']
    maxlen = config['maxlen']


    
    # nb_neurons_bisltm_dense = config['nb_neurons_bisltm_dense']
    
    #Inputs
    # In_text = tf.keras.layers.Input(shape = (), dtype=tf.string, name='text')
    
    In_audio = tf.keras.layers.Input(shape = input_shape, name='audio')



    
    ### Text part
    
    
    vocab_size, nb_dimensions = embedding_matrix.shape
    
    In_text = tf.keras.layers.Input(shape=(maxlen,))
    
    embedding_layer = Embedding(vocab_size, nb_dimensions, weights=[embedding_matrix], 
                                input_length = maxlen, trainable=trainable)(In_text)
    
    BiLSTM_Layer = Bidirectional(LSTM(nb_neurons_bisltm_text))(embedding_layer)
    
    
    X_text = tf.keras.layers.Dropout(rate=dropout_text)(BiLSTM_Layer)

    

    #audio part
    Bi_LSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nb_neurons_bisltm_audio,
                                                                  return_sequences=True,
                                                                  kernel_regularizer=tf.keras.regularizers.L2(l2=0.5)))
    #Att = tf.keras.layers.Attention()
    f0 = tf.keras.layers.Flatten()
    dr0 = tf.keras.layers.Dropout(rate=dropout_audio)

    X_seq = Bi_LSTM(In_audio[:,:,:,0])
    #X_seq = Att([X_seq,X_seq])
    X_seq = f0(X_seq)
    X_seq = dr0(X_seq)

    #cnn part for audio. 
    
    conv1 = tf.keras.layers.Conv2D(filters = conv1_filters, kernel_size = conv1_kernel, 
                                    strides = strides, padding = 'valid')
    bn1 = tf.keras.layers.BatchNormalization()
    a1 = tf.keras.layers.Activation(hidden_activation)
    dr1 = tf.keras.layers.Dropout(rate=dropout_audio)
    mp1 = tf.keras.layers.MaxPool2D(pool_size=pool1_size,strides=strides,
                                    padding='valid')
    conv2 = tf.keras.layers.Conv2D(filters = conv2_filters, kernel_size = conv2_kernel, 
                                    strides = strides, padding = 'valid')
    bn2 = tf.keras.layers.BatchNormalization()
    a2 = tf.keras.layers.Activation(hidden_activation)
    dr2 = tf.keras.layers.Dropout(rate=dropout_audio)
    mp2 = tf.keras.layers.MaxPool2D(pool_size=pool2_size,strides=strides,
                                    padding='valid')
    f = tf.keras.layers.Flatten()

    X = conv1(In_audio)
    X = bn1(X)
    X = a1(X)
    X = dr1(X)
    X = mp1(X)
    X = conv2(X)
    X = bn2(X)
    X = a2(X)
    X = dr2(X)
    X = mp2(X)
    X = f(X)

    X_fin = tf.keras.layers.Concatenate()([X_text,X_seq,X])
    X_fin = tf.keras.layers.Dense(units = nb_neurons_dense)(X_fin)
    X_fin = tf.keras.layers.BatchNormalization()(X_fin)
    X_fin = tf.keras.layers.Activation(hidden_activation)(X_fin)
    X_fin = tf.keras.layers.Dropout(rate=dropout_final)(X_fin)



    out = tf.keras.layers.Dense(units = nb_outputs, activation = tf.nn.softmax)(X_fin)
    model = tf.keras.Model(inputs=[In_text,In_audio], outputs=out)
    return model



class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types for saving in json"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)


