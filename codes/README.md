## Speech synthesis

The `speech_synthesis.py` script is a simple code that runs through the sentences in the file [allDebates_withAnnotations_all_timestamps.csv](https://github.com/rafamestre/Multimodal-USElecDeb60To16/blob/main/Multimodal%20ElecDeb60To16/allDebates_withAnnotations_all_timestamps.csv) and produces artificial utterances using the Zira and Mark voices from Microsoft at a rate of 200 words per minute. The script creates a series of folders for each debate, named '1_1960', '2_1960', ..., '24_1996', '25_2000', ..., '43_2016'. Inside of each folder, it creates *.wav* files for each individual utterance, in the following format: "d[dd]y[yyyy]n[nnnn]", where [dd] represents the debate ID, [yyyy] the year of the debate, [nnnn] the sentence ID. For instance: d09y1980n0686.

The paramters in the script can be modified to select a different rate of speech, as well as other voices found in the system, if they are installed. We provide the artificial utterances for both the Zira and Mark voices at 200 words per minute in the Zenodo repository.

## Extracting audio features

The script `extract_audio_features.py` extracts the audio features from a folder where the utterances have been saved. For instance:

```python
python extract_audio_features.py --audio-dir ../enUS_ZiraM/rate200 --data-dir "../Multimodal ElecDeb60To16" --filename allDebates_withAnnotations_all_timestamps.csv
```

will run the script using the audio files in the folder `enUS_ZiraM/rate200` from the Microsoft Zira voice. It will use the file with the annotations and timestamps ([allDebates_withAnnotations_all_timestamps.csv](https://github.com/rafamestre/Multimodal-USElecDeb60To16/blob/main/Multimodal%20ElecDeb60To16/allDebates_withAnnotations_all_timestamps.csv)) found in the [Multimodal ElecDeb60To16](https://github.com/rafamestre/Multimodal-USElecDeb60To16/tree/main/Multimodal%20ElecDeb60To16) folder.

The folder `enUS_ZiraM/rate200` must contain the same folder and file structure that results from calling the script `speech_synthesis.py`, as described above. That is, it should contain a series of folders for each debate, named '1_1960', '2_1960', ..., '24_1996', '25_2000', ..., '43_2016'. Inside of each folder, there should be *.wav* files for each individual utterance, in the following format: "d[dd]y[yyyy]n[nnnn]", where [dd] represents the debate ID, [yyyy] the year of the debate, [nnnn] the sentence ID. For instance: d09y1980n0686.

The script will loop through each one of the utterances, extract the audio features (MFCCs, spectral centroids, spectral bandwidth, spectral rolloff, spectral contrast, and chroma) and save them in a pickle object called `df_audio_features.pkl` in the folder specified by the --audio-dir argument. Check out the paper for more information about the audio features.

We provide the `df_audio_features.py.pkl` files in the Zenodo repository, not here, so you don't need to run this code to get the audio features of the artificial voices. Unfortunately, we cannot provide the utterances from the original videos to obtain the audio features due to copyright reasons, but we provide the `df_audio_features.py.pkl` of post-processed audio features in the Zenodo repository.

## Runing hyperparameter tuning

To perform hyperparameter tuning, we use the module Ray[tune], which is a Python library for fast hyperparameter tuning at scale. You can run a command like the following:

```shell
python hyperparameter_tuning.py --samples 50 --max-epochs 40 --num-cpus 2 
    --num-gpus 1 --model audio --balanced 
    --name audiomodel_balanced --best-metric acc
```

This code will sample 50 times in the hyperparameter space (see the paper or the code itself for the ranges) for 40 epochs, using 2 CPUs and 1 GPU as resources. It will run the audio-only model with a balanced dataset, that is, balancing the "argument" and "other" classes. If you want to perform the tuning with the full dataset, you can remove the --balanced flag. Likewise, you can use other models like the text-only models based on BERT and Bi-LSTM using 'bert' and 'bilstm' as the --model parameter; the multimodal model based on BERT with 'multimodal'; or the multimodal based on Bi-LSTM for text using 'multimodal2'. You can change the name of the model (and the folder with its results) with the argument --name. (otherwise, it might overwrite). To select the best model during hyperparameter tuning, you can use the --best-metric argument. In this case, it will maximise the accuracy, but you can also choose to minimse the loss with the value 'loss' (not recomended if you compare different architectures), or maximise the area under the curve with 'auc'.

You can also implement skipping audio features, like we did for our ablation study, by using the argument --skip-audio-feature and one of the following values: 'mfccs', 'centroids', 'rolloff', 'chroma', 'contrast', 'bandwitdh'. Only eliminating one feature, and not multiple, is implemented.

You can select fractional dataset with the argument --fract-data. For instance, to keep only 10% of the dataset for tuning, you can use '--fract-data 0.1'. Default is 1 for 100%.

## Running an optimised model

If you don't want to do hyperparameter tuning anymore because you've already identified a configuration that's optimal, you can use the same code to train the model with several replicates and get all the metrics that we report, with averages, standard deviation, plots, etc. To do that, you can run a code like the following:

```shell
python hyperparameter_tuning.py --model audio 
    --config-file ./results/audio_50acc_balanced/2022-6-21_20-9-32_best_config_acc.json 
    --optimize False --max-epochs 60 --balanced --nb-runs 5
```

Here, you will run a balanced audio model using the configuration file of the best model found before. The key argument is --optimize, which should be set to False to indicate that hyperparamter optimisation will NOT be done here. The model will be trained for 60 epochs 5 times, and statistics will be calculated out of the runs.

## Split original audio

The `split_original_audio.py` script runs through all the sentences in the [allDebates_withAnnotations_all_timestamps.csv](https://github.com/rafamestre/Multimodal-USElecDeb60To16/blob/main/Multimodal%20ElecDeb60To16/allDebates_withAnnotations_all_timestamps.csv) file, which contain the timestamps, and simply splits the utterances from the original video as individual utterances.

Unfortunately, due to copyright reasons, we cannot make freely available the original videos, so this function is only provided for transparency. 
