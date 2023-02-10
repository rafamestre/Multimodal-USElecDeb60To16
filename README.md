# Multimodal USElecDeb60To16

This is the accompanying dataset, codes and supplementary information of the paper **"Augmenting pre-trained language models with audio feature embedding for argumentation mining in political debates"**, published at the *Findings of the 17th conference on European chapter of the Association for Computational Linguistics* (EACL) in 2023.

This works augments an already available dataset for argumentation mining of presidential debates (called the USElecDeb60To16 dataset, which can be found here: https://github.com/ElecDeb60To16/Dataset) with timestamped audio files. The original dataset was used for argumentation mining, and thus in this paper we explored multimodal argumentation mining, using both the text and audio.

## Citation

Please, if you use our dataset or find our paper useful, you can cite it in the following way:

>    Mestre, R., Middleton, S. E., Ryan, M., Gheasi, M., Norman, T. J. & Zhu, J. (2023, May). Augmenting pre-trained language models with audio feature embedding for argumentation mining in political debates. In Findings of the 17th conference on European chapter of the Association for Computational Linguistics (EACL).

```
@inproceedings{mestre2023augmenting,
  title={Augmenting pre-trained language models with audio feature embedding for argumentation mining in political debates},
  author={Mestre, R. and Middleton, S. E. and Ryan, M. and Gheasi, M. and Norman, T. J. and Zhu, J.},
  booktitle={Findings of the 17th conference on European chapter of the Association for Computational Linguistics (EACL)},
  year={2023},
  month={May},
  publisher = "Association for Computational Linguistics",
  url = "",
  pages = "",
}
```

And reference the dataset itself in this repository:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7628465.svg)](https://doi.org/10.5281/zenodo.7628465)

```
@dataset{mestre2023_multimodal,
  author       = {Rafael Mestre and
                  Stuart E. Middleton and
                  Matt Ryan and
                  Masood Gheasi and
                  Timothy J. Norman and
                  Jiatong Zhu},
  title        = {rafamestre/Multimodal-USElecDeb60To16: v1.0.0},
  month        = feb,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.7628465},
  url          = {https://doi.org/10.5281/zenodo.7628465}
}
```

## Related publications

There are several other works that have served as inspiration or have tackled the under-research field of multimodal argumentation mining. We give a brief overview here and suggest reading them:

> Lippi, M., Torroni, P., Argument Mining from Speech: Detecting Claims in Political Debates, 30th AAAI Conference on Artificial Intelligence, Phoenix, Arizona, 2016.

This was the first paper (as far as our knowledge goes) that attemped multimodal argumentation mining using text and audio, with a dataset from UK elections (UK 2015 Political Election corpus), which can be found here: http://argumentationmining.disi.unibo.it/aaai2016.html.

> Haddadan, S., Cabrio, E., & Villata, S. (2019, July). Yes, we can! mining arguments in 50 years of US presidential campaign debates. In ACL 2019-57th Annual Meeting of the Association for Computational Linguistics (pp. 4684-4690).

This is the original paper from the USElecDeb60To16 dataset, in which the authors annotated for arguments (premise and claims) the whole set of US televised (vice)presidential debates from 1960 to 2016, obtaining almost 30,000 annotated sentences. Again, the dataset can be found here: https://github.com/ElecDeb60To16/Dataset

>    Mestre, R., Milicin, R., Middleton, S. E., Ryan, M., Zhu, J., & Norman, T. J. (2021, November). M-Arg: Multimodal Argument Mining Dataset for Political Debates with Audio and Transcripts. In Proceedings of the 8th Workshop on Argument Mining (pp. 78-88).

This is our previous work, in which we used the transcripts and audio from the 2020 presidential debates between Donald Trump, Joe Biden, Kamala Harris and Mike Pence to attempt relational multimodal argumentation mining. We used crowdsourcing techniques to create the M-Arg dataset, annotated for relational argumentation (support/attack/neither between sentences). For the audio pipeline, we used deep learning techniques based on convolutional neural networks (CNN) and bidirectional lost short-term memory (Bi-LSTM) arrays to extract audio features. To our knowledge, this was the first time that multimodal models based on deep learning techniques for the audio had been used in the field. Our dataset can be found here: https://github.com/rafamestre/m-arg_multimodal-argumentation-dataset

> Mancini, E., Ruggeri, F., Galassi, A., & Torroni, P. (2022, October). Multimodal Argument Mining: A Case Study in Political Debates. In Proceedings of the 9th Workshop on Argument Mining (pp. 158-170).

This is an excellent work in which the authors presented a comprehensive comparison between multimodal datasets and architectures for argumentation mining, comparing the first work by Lippi and Torroni (2016) and our M-arg dataset and architecture (2021), as well as their own dataset, MM-USElecDeb60to16. Their dataset is also based on the USElecDeb60To16 data provides audio timestamps using the same method we use in the current paper and the M-Arg dataset. Both the present work and this one were prepared concurrently, and therefore both datasets can be considered equivalent. Out Multimodal USElecDeb60to16 is slightly larger than theirs, since we managed to fix some audio syncronisation issues, keeping a larger portion of the original dataset. We have not compared both datasets due to their significant size, but we're certain that both are of high quality. Their dataset can be found at: https://github.com/federicoruggeri/multimodal-am/

## Main results

In this paper, we  explored the possibilities of using audio to detect arguments with multimodal machine learning models in a dataset of US presidential debates that was annotated for arguments. We found that, generally, BERT-based text-only models outperformed all models in the original dataset, but multimodal models can improve performance when the datasets are small and BERT encodings start performing poorly, in both balanced and unbalanced versions of the data. Multimodal models are therefore an alternative that could be used to improve classifications of arguments when data is scarce. To further investigate the reasons for these improvements, we ran  audio-only models using artificially generated voices of male and female genders. Although we did not find a significant difference in the performance with the artificial voices (only a slight preference toward female), we find that in the most difficult scenarios (balanced and small datasets), the models with artificial voices outperform those with the original audio from the debates. Moreover, we perform an ablation study and we find that removing certain features like MFCCs can improve the performance of the models. We recognize that these features are commonly used to distinguish between speakers, so irrelevant characteristics of speakers might be influencing the capacity of the models to accurately classify arguments. However, all these features are highly correlated with one another, so further work should investigate which features are more independent from the speakers themselves or if they can be normalized before being fed into the network. These results should be compared with artificially generated voices, which could have different accents or speech rates to understand how those features can influence the classification of arguments.

Read more in the [paper](https://github.com/rafamestre/Multimodal-USElecDeb60To16/blob/main/mestre_2023_MultimodalUSElecDeb60to16.pdf), included in this repository.

## Subfolder structure

The folders in this repository contain all relevant information to reproduce the results of our paper and use our dataset.

1. The "*alignment*" folder contains all the alignment files with all the timestamps per sentence, as well as the parameters used for the alignment tool. There is a file "*alignment_problems.xlsx*" which reports all our decisions during the alignment process, such as modification of the original dataset ElecDeb60To16 by Haddadan *et al*. (2019).

2. The folder "*results*" contains all the results from training our models with original and balanced datasets, as well as fractional subsets of 50%, 20% and 10%. We also include the training with artificial voices and from the ablation study. Each folder contains the parameters used by the model, confusion matrices of each run (5 runs per model), loss value vs epoch plots, training history with validation metrics, and precision/recall/F-score metrics for each run, as well as the average values. **Note**: 'multimodal' refers to the BERT-based multimodal model, whereas 'multimodal2' refers to the BiLSTM-based multimodal model. 

3. The folder "*Multimodal ElecDeb60To16*" contains the original dataset released by Haddadan *et al*. (2019) as well as our audio-enhanced version with the timestamps and certain modifications to fix typos or mixed sentences that we discussed in the paper.

4. The folders "*enUS_MarkM*" and "*enUS_ZiraM*" contain the artificially generated utterances of each sentence (said by candidates) in the dataset. 

5. The folder "*Original*" is empty, as it would contain the original utterances from the candidates after being split. For copyright reasons, we can't provide them, but we provide scripts to download the video and do the splitting. We do provide, however, a Python pickle file with the extracted audio features of the utterances, called "*df_audio_features.pkl*". 

6. The folder "*videos*" would contain the videos used for the alignment and feature extraction process. Due to copyright reasons, we cannot share them, but we provide the file *YouTubeLinks.csv* with the original sources. Feel free to email the authors about obtaining the videos. There is a subfolder called "*plain*" with the transcripts of the debates in "plain" format (that is, one line per sentence without speaker), which is used by the *aeneas* alignment tool.

7. The folder "*codes*" contains all the Python scripts that we used in this work. They range from the master code for distributed hyperparameter tuning to the scripts that generate the artificial voices or split the audio into utterances. Requirement files are included.


## Running the models and performing hyperparameter tuning

To perform hyperparameter tuning, we use the module Ray[tune], which is a Python library for fast hyperparameter tuning at scale. You can run a command like the following:

```shell
python hyperparameter_tuning.py --samples 50 --max-epochs 40 --num-cpus 2 
    --num-gpus 1 --model audio --balanced 
    --name audiomodel_balanced --best-metric acc
```

This code will sample 50 times in the hyperparameter space (see the [paper](https://github.com/rafamestre/Multimodal-USElecDeb60To16/blob/main/mestre_2023_MultimodalUSElecDeb60to16.pdf) or the code itself for the ranges) for 40 epochs, using 2 CPUs and 1 GPU as resources. It will run the audio-only model with a balanced dataset, that is, balancing the "argument" and "other" classes. If you want to perform the tuning with the full dataset, you can remove the --balanced flag. Likewise, you can use other models like the text-only models based on BERT and Bi-LSTM using 'bert' and 'bilstm' as the --model parameter; the multimodal model based on BERT with 'multimodal'; or the multimodal based on Bi-LSTM for text using 'multimodal2'. You can change the name of the model (and the folder with its results) with the argument --name. (otherwise, it might overwrite). To select the best model during hyperparameter tuning, you can use the --best-metric argument. In this case, it will maximise the accuracy, but you can also choose to minimse the loss with the value 'loss' (not recomended if you compare different architectures), or maximise the area under the curve with 'auc'.

You can also implement skipping audio features, like we did for our ablation study, by using the argument --skip-audio-feature and one of the following values: 'mfccs', 'centroids', 'rolloff', 'chroma', 'contrast', 'bandwitdh'. Only eliminating one feature, and not multiple, is implemented.

You can select fractional dataset with the argument --fract-data. For instance, to keep only 10% of the dataset for tuning, you can use '--fract-data 0.1'. Default is 1 for 100%.

If you don't want to do hyperparameter tuning anymore because you've already identified a configuration that's optimal, you can use the same code to train the model with several replicates and get all the metrics that we report, with averages, standard deviation, plots, etc. To do that, you can run a code like the following:

```shell
python hyperparameter_tuning.py --model audio 
    --config-file ./results/audio_50acc_balanced/2022-6-21_20-9-32_best_config_acc.json 
    --optimize False --max-epochs 60 --balanced --nb-runs 5
```

Here, you will run a balanced audio model using the configuration file of the best model found before. The key argument is --optimize, which should be set to False to indicate that hyperparamter optimisation will NOT be done here. The model will be trained for 60 epochs 5 times, and statistics will be calculated out of the runs.

Read more in the [codes](https://github.com/rafamestre/Multimodal-USElecDeb60To16/tree/main/codes) folder.

## Recovering audio features

**NOTE:** These audio features are not uploaded in this GitHub repository but in our Zenodo repository. 

Audio features are provided as a Python pickle object that contains a dataframe with the following columns: 

* 'ID': contains the ID of the sentence, in the following format "d[dd]y[yyyy]n[nnnn]", where [dd] represents the debate ID, [yyyy] the year of the debate, [nnnn] the sentence ID. For instance: d09y1980n0686.
* 'filepath': contains the filepath of the audio file. For instance: enUS_ZiraM/rate200/10_1984/d10y1984n0032.wav. 
* 'audio_features': contains all the audio features, with shape (45, 97), as they were appended for the audio models.
* The remaining columns, 'mfccs', 'spectral_centroids', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_contrast' and 'chroma_ft' are self-explainatory: they are the individual audio features of each sentence.

Due to size limitations of GitHub, we cannot upload the files here. Please, refer to the Zenodo archive of this repository, where we have added the files in a different version. 

They can be read into Python with Pandas:

```
df = pd.read_pickle(filepath)
```

The audio features are provided for the artificial Zira and Mark voices (for these ones, also the individual utterances are provided in their respective folders [only on Zenodo archive], so audio features can be extracted manually if you wish to), and for the original debates, only their features (for copyright reasons, we cannot provide the original video or audio of the debates at all).
