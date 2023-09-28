import argparse
from collections import defaultdict
from datasets import load_dataset, Dataset
from itertools import chain
import multiprocessing
from multiprocessing import Pool
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import pickle

class TextProcessor:

    def __init__(self, data):
        self.data = data
        self.bow = None
        self.word_to_index = None
        self.index_to_word = None
        self.lemmatized_sentences = None
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')

    def __str__(self):
        """String representation of TextProcessor"""
        return f'TextProcessor(len(data)={len(self.data)})'


    def get_wordnet_pos(self, word):
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    def convert_to_bag_of_words(self, list_of_lists, min_freq=10, max_freq_ratio=0.05):
        # Your existing implementation here
        word_freq = defaultdict(int)
        for lst in list_of_lists:
            for word in lst:
                word_freq[word] += 1
        max_freq = len(list_of_lists) * max_freq_ratio
        vocabulary = {word for word, count in word_freq.items() if min_freq <= count < max_freq}
        word_to_index = {word: i for i, word in enumerate(vocabulary)}
        index_to_word = {i: word for word, i in word_to_index.items()}
        num_lists = len(list_of_lists)
        vocab_size = len(vocabulary)
        bag_of_words = [[0] * vocab_size for _ in range(num_lists)]
        self.lemmas = []
        for i, lst in enumerate(list_of_lists):
            lemma = []
            for word in lst:
                lemma = [word for word in lst if word in word_to_index]
                if word in word_to_index:
                    index = word_to_index[word]
                    bag_of_words[i][index] += 1
                    
            self.lemmas.append(lemma)

        self.bow, self.word_to_index, self.index_to_word = bag_of_words, word_to_index, index_to_word

    def extract_important_words(self, tfidf_vector, feature_names):
        # Your existing implementation here
        coo_matrix = tfidf_vector.tocoo()
        sorted_items = sorted(zip(coo_matrix.col, coo_matrix.data), key=lambda x: (x[1], x[0]), reverse=True)

        return self.extract_topn_from_vector(feature_names, sorted_items)

    def extract_topn_from_vector(self, feature_names, sorted_items, topn=20):
        sorted_items = sorted_items[:topn]

        score_vals = []
        feature_vals = []

        for idx, score in sorted_items:
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])

        results = {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]] = score_vals[idx]

        return results

    # def lemmatize_sentences(self):
    #     lemmatizer = WordNetLemmatizer()
    #     lemmatized_sentences = []
    #     stop_words = set(stopwords.words('english'))
    #     table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    #     for index, sentence in enumerate(self.data):
    #         if index % 100 == 0:
    #             print(index)
    #         sentence = sentence.translate(table).lower().replace("  ", " ")

    #         words = word_tokenize(sentence)
    #         lemmatized_words = [lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) for word in words if word not in stop_words and lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) != '' ]
    #         lemmatized_words = [i for i in lemmatized_words if len(i) >= 3 and (not i.isdigit()) and ' ' not in i]
    #         lemmatized_sentences.append(lemmatized_words)

    #     self.lemmatized_sentences = lemmatized_sentences
    

    def worker(self, data_chunk):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        lemmatized_chunk = []

        for sentence in data_chunk:
            sentence = sentence.translate(table).lower().replace("  ", " ")
            words = word_tokenize(sentence)
            lemmatized_words = [lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) for word in words if word not in stop_words and lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) != '' ]
            lemmatized_words = [i for i in lemmatized_words if len(i) >= 3 and (not i.isdigit()) and ' ' not in i]
            lemmatized_chunk.append(lemmatized_words)

        return lemmatized_chunk

    def lemmatize_sentences(self):
        num_processes = multiprocessing.cpu_count()
        pool = Pool(num_processes)
        data_chunks = np.array_split(self.data, num_processes)
        results = pool.map(self.worker, data_chunks)
        pool.close()
        pool.join()
        #print(results)
        self.lemmatized_sentences = [j for i in results for j in i]
        
        print(len(results), len(results[0]))

    def process(self):
        self.lemmatize_sentences()
        self.convert_to_bag_of_words(self.lemmatized_sentences)