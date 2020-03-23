import preprocessing.preprocesstools as ppt
import preprocessing.sources.newsgroup as source
import multiprocessing as mp


dataset = source.retrieve()

min_words_in_doc = 20 # Minimum number of words in a document

# Minimum % of documents in wich each word of the corpus must appear
min_word_occurences = 0.05  

# Maximum % of documents in wich each word of the corpus must appear
max_word_occurences = 0.95

cores = mp.cpu_count()
stopwords_extension = ['from', 'subject', 're', 'edu', 'use']

preprocessed = ppt.preprocess_multiprocess(
    cores, dataset, min_words_in_doc, min_word_occurences, max_word_occurences, stopwords_extension)

ppt.save(preprocessed, "ppTest.json")
