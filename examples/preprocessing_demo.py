import os
os.chdir(os.path.pardir)
from dataset.dataset import Dataset
import preprocessing.sources.reuters as source
from preprocessing.pipeline_handler import Pipeline_handler
import multiprocessing as mp


dataset = source.retrieve()

# All parameters defaults are True
pipeline_handler = Pipeline_handler(dataset, num_processes=mp.cpu_count(), display_progress=True,
                                    min_words_for_doc=2, words_min_freq=0.01,
                                    words_max_freq=1, stopwords='english')

preprocessed = pipeline_handler.preprocess()

preprocessed.save("dataset_folder")

