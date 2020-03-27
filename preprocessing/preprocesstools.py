import json
import gensim
import re
import spacy
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import multiprocessing as mp

from pathlib import Path


stop_words = stopwords.words('english')


# Saves metadata in json serialized format
def _save_metadata(metadata, file_name):
    with open(file_name, 'w') as outfile:
        json.dump(metadata, outfile)
    return True


# Saves the corpus, a document for each line
def _save_corpus(data, file_name):
    with open(file_name, 'w') as outfile:
        for element in data:
            outfile.write("%s\n" % " ".join(element))
    return True


# Saves the labels, each line contains the labels of a single document
def _save_labels(data, file_name):
    with open(file_name, 'w') as outfile:
        for element in data:
            outfile.write("%s\n" % json.dumps(element))
    return True


# Saves the vocabulary in list format
def _save_vocabulary(vocabulary, file_name):
    with open(file_name, 'w') as outfile:
        for word in vocabulary:
            outfile.write("%s\n" % word)


# Saves all corpus, vocabulary and metadata
def save(data, path):
    Path(path).mkdir(parents=True, exist_ok=True)
    if "corpus" in data:
        _save_corpus(data["corpus"], path+"/corpus.txt")
    if "vocabulary" in data:
        _save_vocabulary(data["vocabulary"], path+"/vocabulary.txt")
    if "labels" in data:
        _save_labels(data["labels"], path+"/labels.txt")
    if "metadata" in data:
        _save_metadata(data["metadata"], path+"/metadata.json")
    return True


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


def remove_stopwords(corpus, *_):
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
    vocabulary = {}
    for document in corpus:
        for word in document:
            if not word in vocabulary:
                vocabulary[word] = True
    return list(vocabulary.keys())



# Execute a custom preprocess routine in multicore
def preprocess_multiprocess(num_proc, dataset, **args):
    # Default: execute all steps
    # inizialize variables to execute all steps
    min_words_for_doc = 0
    words_min_freq = 0
    words_max_freq = len(dataset["corpus"])
    stop_words_extension = []
    rm_punctuation = True
    rm_stopwords = True
    lemmatize = True
    rm_words = True

    # for each step:
    # step_name = True to execute the step, False otherwise
    if "remove_punctuation" in args:
        rm_punctuation = args["remove_punctuation"]
    if "remove_stopwords" in args:
        rm_stopwords = args["remove_stopwords"]
    if "lemmatize" in args:
        lemmatize = args["lemmatize"]
    if "words_min_freq" in args:
        words_min_freq = args["words_min_freq"]
    if  "words_max_freq" in args:
        words_max_freq = args["words_max_freq"]
    if "remove_words" in args:
        rm_words = args["remove_words"]
    if "min_words_for_doc" in args:
        min_words_for_doc = args["min_words_for_doc"]
    if "stop_words_extension" in args:
        stop_words_extension = args["stop_words_extension"]
    
    stop_words.extend(stop_words_extension)
    corpus = dataset["corpus"]
    categories = []
    if "doc_labels" in dataset:
        categories = dataset["doc_labels"]
    print("Creating pool")
    pool = create_pool(num_proc)
    print("Pool created\nPreprocess initialization\n\n")

    # Execute steps
    if rm_punctuation:
        print("  removing punctuation")
        corpus = _multiprocess(pool, num_proc, remove_punctuation, corpus)
        print("  Punctuation removed\n\n")

    corpus_bag = list(create_bags_of_words(corpus))

    if rm_stopwords:
        print("  Removing Stopwords")
        corpus_bag = _multiprocess(pool, num_proc, remove_stopwords, corpus_bag)
        print("  Stopwords removed\n\n")

    if lemmatize:
        print("  lemmatizing")
        nlp = spacy.load('en', disable=['parser', 'ner'])
        corpus_bag = _multiprocess(pool, num_proc, lemmatization, corpus_bag, [
                                    nlp, ['NOUN', 'ADJ', 'VERB', 'ADV']])
        print("  Lemmatized\n\n")
    
    if rm_words:
        print("  Removing words that are present in less than " +
            str(words_min_freq)+" documents")
        print(" and im more than "+str(words_min_freq)+" documents")
        to_remove = words_to_remove(
            corpus_bag, words_min_freq, words_max_freq)
        corpus_bag = _multiprocess(
            pool, num_proc, remove_words, corpus_bag, to_remove)
        print("  Words removed\n\n") 
    
    print("  Removing documents with less then " +
          str(min_words_for_doc)+" words")
    result = remove_docs(corpus_bag, min_words_for_doc, categories)
    print("  Documents removed\n\nPreprocess done!")

    pool.close()
    return result

    
# Execute a standard preprocess routine in multicore
def preprocess_multiprocess_standard(num_proc, dataset, min_words_for_doc, words_min_freq, words_max_freq, stop_words_extension=[]):
    return preprocess_multiprocess(
    num_proc,
    dataset,
    min_words_for_doc = min_words_for_doc,
    words_min_freq = words_min_freq,
    words_max_freq =  words_max_freq,
    stop_words_extension = stop_words_extension)


# Execute a custom preprocess routine
def preprocess(dataset, **args):
    # Default: execute all steps
    # inizialize variables to execute all steps
    min_words_for_doc = 0
    words_min_freq = 0
    words_max_freq = len(dataset["corpus"])
    stop_words_extension = []
    rm_punctuation = True
    rm_stopwords = True
    lemmatize = True
    rm_words = True

    # for each step:
    # step_name = True to execute the step, False otherwise
    if "remove_punctuation" in args:
        rm_punctuation = args["remove_punctuation"]
    if "remove_stopwords" in args:
        rm_stopwords = args["remove_stopwords"]
    if "lemmatize" in args:
        lemmatize = args["lemmatize"]
    if "words_min_freq" in args:
        words_min_freq = args["words_min_freq"]
    if  "words_max_freq" in args:
        words_max_freq = args["words_max_freq"]
    if "remove_words" in args:
        rm_words = args["remove_words"]
    if "min_words_for_doc" in args:
        min_words_for_doc = args["min_words_for_doc"]
    if "stop_words_extension" in args:
        stop_words_extension = args["stop_words_extension"]
    
    stop_words.extend(stop_words_extension)
    corpus = dataset["corpus"]
    categories = []
    if "doc_labels" in dataset:
        categories = dataset["doc_labels"]

    # Execute steps
    if rm_punctuation:
        print("  removing punctuation")
        corpus = remove_punctuation(corpus)
        print("  Punctuation removed\n\n")

    corpus_bag = list(create_bags_of_words(corpus))

    if rm_stopwords:
        print("  Removing Stopwords")
        corpus_bag = remove_stopwords(corpus_bag)
        print("  Stopwords removed\n\n")

    if lemmatize:
        print("  lemmatizing")
        nlp = spacy.load('en', disable=['parser', 'ner'])
        corpus_bag = lemmatization(corpus_bag,
            [nlp, ['NOUN', 'ADJ', 'VERB', 'ADV']])
        print("  Lemmatized\n\n")
    
    if rm_words:
        print("  Removing words that are present in less than " +
            str(words_min_freq)+" documents")
        print(" and im more than "+str(words_min_freq)+" documents")
        to_remove = words_to_remove(
            corpus_bag, words_min_freq, words_max_freq)
        corpus_bag = remove_words(corpus_bag, to_remove)
        print("  Words removed\n\n") 
    
    print("  Removing documents with less then " +
          str(min_words_for_doc)+" words")
    result = remove_docs(corpus_bag, min_words_for_doc, categories)
    print("  Documents removed\n\nPreprocess done!")

    return result

    
# Execute a standard preprocess routine in multicore
def preprocess_standard(dataset, min_words_for_doc, words_min_freq, words_max_freq, stop_words_extension=[]):
    return preprocess(
    dataset,
    min_words_for_doc = min_words_for_doc,
    words_min_freq = words_min_freq,
    words_max_freq =  words_max_freq,
    stop_words_extension = stop_words_extension)