import preprocessing.preprocesstools as ppt
import preprocessing.sources.newsgroup as source
import multiprocessing as mp


dataset = source.retrieve()

min_words_in_doc = 1  # Minimum number of words in a document

# Minimum % of documents in wich each word of the corpus must appear
min_word_occurences = 0.01

# Maximum % of documents in wich each word of the corpus must appear
max_word_occurences = 0.90

cores = mp.cpu_count()

stopwords_extension = ['from', 'subject', 're', 'edu', 'use']

# All parameters defaults are True
preprocessed = ppt.preprocess_multiprocess(
    cores, dataset, min_words_for_doc = min_words_in_doc,
    lemmatize = True,
    remove_stopwords = True,
    remove_punctuation = True,
    remove_words = True,
    words_min_freq = min_word_occurences,
    words_max_freq =  max_word_occurences)

ppt.save(preprocessed, "dataset")
