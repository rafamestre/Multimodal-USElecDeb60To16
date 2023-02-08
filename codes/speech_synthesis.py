# -*- coding: utf-8 -*-
"""
Created on Fri May 13 17:34:33 2022

@author: Rafael Mestre, r.mestre@soton.ac.uk

https://github.com/rafamestre/Multimodal-USElecDeb60To16
"""

import pyttsx3

from playsound import playsound
import sys
import os

import pandas as pd
from pathlib import Path

from hyperparameter_util_functions import clean_text_tags





# How to add voices to Windows
# https://www.thewindowsclub.com/unlock-extra-text-to-speech-voices-in-windows
# https://windowsreport.com/windows-10-language-pack-error-0x800f0954/ for the error
# The UseWUServer registry key needs to be set to 0 and then restarted to install the packages
# They need to appear in C:\Windows\Speech_OneCore\Engines\TTS

def print_available_voices(voices):
    
    for voice in voices: 
        print("Voice:") 
        print("ID: %s" %voice.id) 
        print("Name: %s" %voice.name) 
        print("Age: %s" %voice.age) 
        print("Gender: %s" %voice.gender) 
        print("Languages Known: %s" %voice.languages)
        print("\n")



def text_to_speech(row, synthesiser, filepath, column_text = 'Speech', column_ID = 'ID'):
    
    text = row[column_text]
    ID = row[column_ID]
    
    # synthetiser.say(text)
    # print(Path(filepath, ID+'.mp3'))
    synthesiser.save_to_file(text,str(Path(filepath, ID+'.mp3')))
    # synthesizer.runAndWait()


def insert_timestamps(df):

    ######INSERT THE TIMESTAMPS INTO THE DATASET
    filepath_videos = Path(r'..\videos')

    df_links = pd.read_csv(Path(filepath_videos,'YoutubeLinks.csv'))
    # df = pd.read_csv(Path(filepath_processed,filename))

    filepath_alignment = Path(r'..\alignment')
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






# Read speech data
filepath_data = Path(r'../Multimodal ElecDeb60To16') #Or change to your own directory
filename = 'allDebates_withAnnotations.csv'
df = pd.read_csv(Path(filepath_data,filename)) 


    

# Insert the timestamps
df_all = insert_timestamps(df)

# Clean the text for tags
df_all = clean_text_tags(df_all)


# Retain only those that have arguments
df_all = df_all[~(df_all['Component'].isnull())]

# And only those that are from candidates
df_all = df_all[df_all['SpeakerType']=='Candidate']




# Define voides (you can add more if you have more installed)
voice_list = ['Microsoft Zira - English (United States)',
              'Microsoft Mark - English (United States)']

voices = pyttsx3.init().getProperty('voices')

# Select speech rate
rate = 200


# Run for each voice and save the utterance
for voice in voices:

    if not voice.name in voice_list:
        continue
    
    print(voice)
    
    voice_id = voice.id
    voice_name = Path(voice_id).stem.split('_')[-1]
    voice_lang = Path(voice_id).stem.split('_')[2]
    
    filepath_voice = Path('..')/Path(voice_lang+'_'+voice_name)

    if not os.path.exists(filepath_voice):
        os.makedirs(filepath_voice)
    
    # Initialise speech synthetiser
    synthesiser = pyttsx3.init()
    synthesiser.setProperty('rate',rate)
    synthesiser.setProperty('voice', voice_id)  
    
    filepath_rate = Path(filepath_voice,'rateee'+str(rate))
    
    if not os.path.exists(filepath_rate):
        os.makedirs(filepath_rate)

    
    for d in df_all['Document'].unique():
        
        filepath_debate = Path(filepath_rate,d)
        
        if not os.path.exists(filepath_debate):
            os.makedirs(filepath_debate)

        df_this = df_all[df_all['Document']==d]
        df_this = df_this[df_this['Annotated'] == True] #Only annotated ones
    
    
        df_this.apply(text_to_speech,args=(synthesiser, filepath_debate), axis=1)
        synthesiser.runAndWait()

        sys.exit()



































