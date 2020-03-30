import gensim
import re
import spacy
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import multiprocessing as mp


def create_pool(n_cpu):
    pool = mp.Pool(processes=n_cpu)
    return pool


# Execute a function on a corpus in multicore
def _multiprocess(pool, n_cpu, function, corpus, arguments=[]):
    corpus_length = len(corpus)
    documents_for_cpu = round(corpus_length/n_cpu) + 1
    tmp_tot = corpus_length
    cumulated_range_interval = 0
    ranges = [0]
    for _ in range(n_cpu):
        if tmp_tot - documents_for_cpu >= 0:
            tmp_tot -= documents_for_cpu
            cumulated_range_interval += documents_for_cpu
            ranges.append(cumulated_range_interval)
        else:
            ranges.append(corpus_length)
    results = [pool.apply(function, args=(
        corpus[ranges[y]:ranges[y+1]], arguments)) for y in range(n_cpu)]
    for y in range(n_cpu):
        corpus[y*documents_for_cpu:y*documents_for_cpu +
               documents_for_cpu] = results[y]
    return corpus


def remove_punctuation(corpus, *_):
    corpus = [re.sub(r'\S*@\S*\s?', '', doc) for doc in corpus]
    corpus = [re.sub(r'\s+', ' ', doc) for doc in corpus]
    corpus = [re.sub(r"\'", "", doc) for doc in corpus]
    return corpus


# convert each document in the corpus from string
#  into a list of words
def create_bags_of_words(corpus):
    for doc in corpus:
        yield(gensim.utils.simple_preprocess(str(doc), deacc=True))


def remove_stopwords(corpus, stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in corpus]


def lemmatization(corpus, arguments):
    result = []
    for document in corpus:
        doc = arguments[0](" ".join(document))
        result.append(
            [token.lemma_ for token in doc if token.pos_ in arguments[1]])
    return result


# Creates a dictionary of words to remove from the documents
def words_to_remove(corpus, min_word_freq, max_word_freq):
    corpus_length = len(corpus)
    minimum = round(min_word_freq*corpus_length)
    maximum = round(max_word_freq*corpus_length)
    words_dict = {}
    for document in corpus:
        word_found_in_article = {}
        for word in document:
            if word in words_dict:
                if not word in word_found_in_article:
                    words_dict[word] += 1
                    word_found_in_article[word] = True
            else:
                words_dict[word] = 1
                word_found_in_article[word] = True
    to_remove = {}
    for key, value in words_dict.items():
        if value <= minimum or value >= maximum:
            to_remove[key] = True
    return to_remove


# Remove from the documents each occurence of the words in input
def remove_words(corpus, words):
    result = []
    for document in corpus:
        result.append([word for word in document if not word in words])
    return result


# Remove documents with less then min_doc words
# from the corpus and create a dictionary with
# extra data about the corpus
def remove_docs(corpus, min_doc = 0, labels = []):
    n = 0
    new_corpus = []
    new_labels = []
    compute_labels = len(labels)>0
    words_mean = 0
    distinct_labels = {}
    for document in corpus:
        document_length = len(document)
        if document_length > min_doc:
            words_mean += document_length
            new_corpus.append(document)
            if compute_labels:
                new_labels.append(labels[n])
                for label in labels[n]:
                    if not label in distinct_labels:
                        distinct_labels[label] = True
            n += 1
    words_document_mean = 0
    if n > 0:
        words_document_mean = round(words_mean/n)
    result = {}
    result["corpus"] = new_corpus
    result["vocabulary"] = get_vocabulary(new_corpus)
    result["labels"] = new_labels
    extra_info = {}
    extra_info["total_documents"] = n
    extra_info["words_document_mean"] = words_document_mean
    extra_info["vocabulary_length"] = len(result["vocabulary"])
    if compute_labels:   
        extra_info["labels"] = list(distinct_labels.keys())
        extra_info["total_labels"] = len(extra_info["labels"])
    result["metadata"] = extra_info
    return result


# Retrieve the vocabulary from the corpus
def get_vocabulary(corpus):
    corpus_length = len(corpus)
    vocabulary = {}
    for document in corpus:
        word_in_document = {}
        for word in document:
            if word in vocabulary:
                if not word in word_in_document:
                    word_in_document[word]= True
                    vocabulary[word] += 1
            else:
                word_in_document[word] = True
                vocabulary[word] = 1
    
    for word in vocabulary:
        vocabulary[word] = round(vocabulary[word]/corpus_length,4)

    return vocabulary
