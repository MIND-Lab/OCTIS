import os

import octis.preprocessing.sources.reuters as source
from octis.preprocessing.pipelinehandler import PipelineHandler
import multiprocessing as mp

os.chdir(os.path.pardir)

dataset = source.retrieve_reuters()

# All parameters defaults are True
pipeline_handler = PipelineHandler(dataset, num_processes=mp.cpu_count(), display_progress=True,
                                   min_words_for_doc=2, words_min_freq=0.01,
                                   words_max_freq=1, stopwords='english')

preprocessed = pipeline_handler.preprocess()

preprocessed.save("dataset_folder")
