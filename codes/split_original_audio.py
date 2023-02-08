# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 16:08:06 2022

@author: Rafael Mestre, r.mestre@soton.ac.uk

https://github.com/rafamestre/Multimodal-USElecDeb60To16

Function that splits the original audio files into utterances.

"""

import sys
import os
import pandas as pd
from pathlib import Path

from moviepy.editor import AudioFileClip
    
# Define filepaths
filepath_data = Path(r'../Multimodal ElecDeb60To16') #Or change to your own directory
filepath_video = Path(r'../videos') #Or change to your own directory
filepath_save = Path(r'../Original') # Videos should be here

# Load dataframe
filename = 'allDebates_withAnnotations_all_timestamps.csv'
df = pd.read_csv(Path(filepath_data,filename)) 

# Retain only those sentences that have arguments
df = df[~(df['Component'].isnull())]

# And only those that are from candidates
df = df[df['SpeakerType']=='Candidate']

# Get a list of all debates
debates = df['Document'].unique()

# And a list of all videos
video_list = os.listdir(filepath_video)


def clip_audio(audio,start, end, buffer = 0):
    """
    This function takes an audio file and start and end timestamps, 
    and returns a subclip of the audio file between the start and end timestamps. 
    
    Parameters:
    audio (AudioFileClip): The audio file to be clipped.
    start (float): The start timestamp of the subclip, in seconds.
    end (float): The end timestamp of the subclip, in seconds.
    buffer (float, optional): The buffer to be added before and after 
        the start and end timestamps, in seconds. The default is 0.
    
    Returns:
    AudioFileClip: The subclip of the audio file between the start 
        and end timestamps.
    """
    
    if not buffer:
        extract = audio.subclip(t_start=(start),t_end=(end))
    
    if start < buffer:
            extract = audio.subclip(t_start=(start),t_end=(end+buffer))
    else:
        try:
            extract = audio.subclip(t_start=(start-buffer),t_end=(end+buffer))
        except:
            extract = audio.subclip(t_start=(start-buffer),t_end=(end))

    return extract

# sys.exit()

for i, d in enumerate(debates):
    print('Doing debate {}'.format(d))
    
    video_file = [s for s in video_list if d in s]
    
    if len(video_file) > 1:
        print('More than one video was found for debate {}. Skipping...'.format(d))
    elif len(video_file) == 0:
        print('No file for debate {} was found.'.format(d))
    
    if not os.path.exists(Path(filepath_save,d)):
        os.makedirs(Path(filepath_save,d))
    else:
        continue

    print('Reading file...')
    audioclip = AudioFileClip(str(Path(filepath_video, video_file[0])))
    print('File read.')
    
    df_this = df[df['Document']==d]
    

    print('Starting audio clipping...')
    for j, row in df_this.iterrows():
        
        sentence_id = row['ID']
        start = row['Begin_s']
        end = row['End_s']
        
    
        #Extract each sentence separately
        sentence = clip_audio(audioclip,start,end)
    
        #Save them
        sentence.write_audiofile(Path(filepath_save, d + '\\' +  sentence_id+'.wav'))

        
    

