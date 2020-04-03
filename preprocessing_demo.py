from dataset.dataset import Dataset
import preprocessing.sources.reuters as source
from preprocessing.pipeline_handler import Pipeline_handler
import multiprocessing as mp


dataset = source.retrieve()

# All parameters defaults are True
pipeline_handler = Pipeline_handler(dataset)

pipeline_handler.multiprocess = True

pipeline_handler.num_proc = mp.cpu_count()

pipeline_handler.display_progress = True

# Minimum number of words in a document
pipeline_handler.min_words_for_doc = 10

# Minimum % of documents in wich each word of the corpus must appear
pipeline_handler.words_min_freq = 0.01

# Maximum % of documents in wich each word of the corpus must appear
pipeline_handler.words_max_freq = 0.9

pipeline_handler.stop_words_extension = ['from', 'subject', 're', 'edu', 'use']


preprocessed = pipeline_handler.preprocess()

preprocessed.save("dataset_folder")
