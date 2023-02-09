# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 12:42:39 2022

@author: Rafael Mestre, r.mestre@soton.ac.uk

https://github.com/rafamestre/Multimodal-USElecDeb60To16

"""

from pathlib import Path
import pandas as pd
import sys
import numpy as np
import pickle
import librosa
from tqdm import tqdm
import argparse



def audio_extract_features_and_save(df,filepath_save):
    
    # Loop through the whole dataframe that extracts the audio features of
    # the sentences
    audio_features = []
    mfccs = []
    spectral_centroids = []
    spectral_bandwidth = []
    spectral_rolloff = []
    spectral_contrast = []
    chroma_ft = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            
            y, sr = librosa.load(row['filepath'])
            mfccs.append(librosa.feature.mfcc(y=y, sr=sr,n_mfcc=25)[2:])
            spectral_centroids.append(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_bandwidth.append(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_rolloff.append(librosa.feature.spectral_rolloff(y=y, sr=sr))
            spectral_contrast.append(librosa.feature.spectral_contrast(y=y, sr=sr))
            chroma_ft.append(librosa.feature.chroma_stft(y=y, sr=sr))
            features = np.concatenate((spectral_centroids[-1], 
                                       spectral_bandwidth[-1], 
                                       spectral_rolloff[-1], 
                                       spectral_contrast[-1], 
                                       chroma_ft[-1], mfccs[-1]),axis=0)
            audio_features.append(features)
        except Exception as e:
            # This is for the case when the audio sentences have 0 duration 
            # (there are some because of the alignment software interracting with certain complicated situations)
            audio_features.append(None)
            mfccs.append(None)
            spectral_centroids.append(None)
            spectral_bandwidth.append(None)
            spectral_rolloff.append(None)
            spectral_contrast.append(None)
            chroma_ft.append(None)

            print("Sentence {} removed from dataset due to faulty audio feature extraction.".\
                  format(row['ID']))
            print(row['filepath'])
            print(e)
            continue
    
    df['audio_features'] = audio_features
    df['mfccs'] = mfccs
    df['spectral_centroids'] = spectral_centroids
    df['spectral_bandwidth'] = spectral_bandwidth
    df['spectral_rolloff'] = spectral_rolloff
    df['spectral_contrast'] = spectral_contrast
    df['chroma_ft'] = chroma_ft
    
    with open(Path(filepath_save,'df_audio_features.pkl'), 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return df






def main():

    parser = argparse.ArgumentParser(description='Audio extraction')
    

    
    
    parser.add_argument('--data-dir', type=str, 
                        help='Directory where the dataset is.', 
                        default=Path('../Multimodal ElecDeb60To16'))

    parser.add_argument('--audio-dir', type=str, 
                        help='Directory where the audio is.', 
                        default=Path('../Original'))

    parser.add_argument('--filename', type=str, 
                        help='Name of the dataset.', 
                        default='allDebates_withAnnotations_all_timestamps.csv')

    args = sys.argv[1:]
    args = parser.parse_args(args)
    
    filepath_data = args.data_dir
    
    filename = args.filename
    
    df = pd.read_csv(Path(filepath_data,filename)) 
    
    
    
    
    df_arguments = df.copy()
    # remove rows that are empty
    df_arguments = df_arguments[~(df_arguments['Component'].isnull())]
    
    
    
    
    # filepath_save = Path(r'C:\Users\rmc1r21\Documents\Argmin22 voices\Original')
    filepath_audio = args.audio_dir
    
    
    
    
    
    
    df_arguments['filepath'] = df_arguments.apply(lambda x: str(Path(filepath_audio,
                                                                 x['Document']+'/'+x['ID']+'.wav')),
                                                  axis=1)
    
    
    df_features = audio_extract_features_and_save(df_arguments[['ID','filepath']],
                                                  filepath_audio)
    
    

if __name__ == "__main__":
    main()

