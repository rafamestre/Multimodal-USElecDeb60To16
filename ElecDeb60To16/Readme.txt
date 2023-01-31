Explanation of the files:

allDebates_withAnnotations_all_timestamps.cvs: final dataset after cleaning and fixing the original dataset and performing the audio alignment process. This is the only file that's used for hyperparameter tuning and the final models.

part_db_candidate.csv: original dataset from Haddadan et al. (2019), with annotations divided in segments or speechs (parts).

pard_db_candidate_fixed.csv: same dataset as above, but after fixing some small mistakes that we found.

sentence_db_candidate.csv: original dataset from Haddadan et al. (2019), with annotations divided into sentences.

sentende_db_candidate_fixed.csv: same dataset as above, but after fixing some small mistakes that we found. This was used to generate the main dataset.

ElectDeb60To16_Guidelines.pdf: annotation guidelines from the authors of the original dataset ElectDeb60To16.