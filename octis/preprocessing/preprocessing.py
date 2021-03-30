import string
from typing import List, Union
from multiprocessing import Pool
from functools import partial
import re
from tqdm.contrib.concurrent import process_map  # or thread_map
from itertools import product

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from octis.dataset.dataset import Dataset

spacy_model_mapping = {'chinese': 'zh_core_web_sm', 'danish': 'nl_core_news_sm', 'dutch': 'nl_core_news_sm',
                       'english': 'en_core_web_sm', 'french': 'fr_core_news_sm', 'german': 'de_core_news_sm',
                       'greek': 'el_core_news_sm', 'italian': 'it_core_news_sm', 'japanese': 'ja_core_news_sm',
                       'lithuanian': 'lt_core_news_sm', 'norwegian': 'nb_core_news_sm', 'polish': 'pl_core_news_sm',
                       'portoguese': 'pt_core_news_sm', 'romanian': 'ro_core_news_sm', 'russian': 'ru_core_news_sm',
                       'spanish': 'es_core_news_sm'}


class Preprocessing:
    def __init__(self, lowercase=True, vocabulary=None, max_features=None, min_df=0.0, max_df=1.0,
                 remove_punctuation=True, punctuation=string.punctuation, remove_numbers=True, lemmatize=True,
                 stopword_list: Union[str, List[str]] = None, min_chars=1, min_words_docs=0, language='english',
                 split=True, verbose=False, strip=True, num_processes=None):
        self.vocabulary = vocabulary
        self.lowercase = lowercase
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.remove_punctuation = remove_punctuation
        self.punctuation = punctuation
        self.lemmatize = lemmatize
        self.language = language
        self.strip = strip
        self.num_processes = num_processes
        self.remove_numbers = remove_numbers

        if self.lemmatize:
            lang = spacy_model_mapping[self.language]
            try:
                self.spacy_model = spacy.load(lang)
            except IOError:
                raise IOError("Can't find model " + lang + ". Check the data directory or download it using the "
                                                           "following command:\npython -m spacy download " + lang)
        self.split = split
        self.verbose = verbose
        if type(stopword_list) == list:
            stopwords = set(stopword_list)
            self.remove_stopwords_spacy = False
        else:
            if 'english' in stopword_list:
                with open('octis/preprocessing/stopwords/english.txt') as fr:
                    stopwords = [line.strip() for line in fr.readlines()]
                    assert stopword_list == language
            else:
                self.remove_stopwords_spacy = True
                assert stopword_list == language

                stopwords = []

        self.stopwords = stopwords
        self.min_chars = min_chars
        self.min_doc_words = min_words_docs
        self.preprocessing_steps = []

    def preprocess_dataset(self, documents_path, labels_path=None):
        docs = [line.strip() for line in open(documents_path, 'r').readlines()]
        if self.num_processes is not None:
            #with Pool(self.num_processes) as p:
            #    docs = p.map(self.simple_preprocessing_steps, docs)
            docs = process_map(self.simple_preprocessing_steps, docs, max_workers=self.num_processes, chunksize=1)
        else:
            docs = self.simple_preprocessing_steps(docs)
        if self.lowercase:
            self.preprocessing_steps.append("lowercase")
        if self.remove_punctuation:
            self.preprocessing_steps.append('remove_punctuation')
        if self.lemmatize:
            self.preprocessing_steps.append('lemmatize')

        vocabulary = self.filter_words(docs)
        print("created vocab")
        #with Pool(self.num_processes) as p:
        #    final_docs, final_labels = p.starmap(self._foo, product(docs, vocabulary, labels_path, repeat=2))
        print(len(vocabulary))
        final_docs, final_labels = [], []
        if labels_path is not None:
            labels = [line.strip() for line in open(labels_path, 'r').readlines()]
            for i, doc, label in zip(range(len(docs)), docs, labels):
                vocab = set(vocabulary)
                new_doc = [w for w in doc.split() if w in vocab]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
                    final_labels.append(label)
                print(i)
        else:
            for doc in docs:
                vocab = set(vocabulary)
                new_doc = [w for w in doc.split() if w in vocab]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
        self.preprocessing_steps.append('filter documents with less than ' + str(self.min_doc_words) + " words")
        if self.verbose:
            print("words filtering done")
        metadata = {"total_documents": len(docs), "vocabulary_length": len(vocabulary),
                    "preprocessing-info": self.preprocessing_steps, "labels": list(set(final_labels)),
                    "total_labels": len(set(final_labels))}
        print("inizio split")
        if self.split:
            if len(final_labels) > 0:
                train, test, y_train, y_test = train_test_split(
                    range(len(final_docs)), final_labels, test_size=0.15, random_state=1, stratify=final_labels)

                train, validation = train_test_split(train, test_size=3 / 17, random_state=1, stratify=y_train)
                partitioned_labels = [final_labels[doc] for doc in train + validation + test]
                partitioned_corpus = [final_docs[doc] for doc in train + validation + test]
                metadata["last-training-doc"] = len(train)
                metadata["last-validation-doc"] = len(validation) + len(train)

                return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata, labels=partitioned_labels)
            else:
                train, test = train_test_split(range(len(final_docs)), test_size=0.15, random_state=1)
                train, validation = train_test_split(train, test_size=3 / 17, random_state=1)

                metadata["last-training-doc"] = len(train)
                metadata["last-validation-doc"] = len(validation) + len(train)
                partitioned_corpus = [final_docs[doc] for doc in train + validation + test]

                return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata, labels=final_labels)
        else:
            Dataset(final_docs, vocabulary=vocabulary, metadata=metadata, labels=final_labels)

    def filter_words(self, docs):
        if self.vocabulary is not None:
            self.preprocessing_steps.append('filter words by vocabulary')
            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")
            vectorizer = TfidfVectorizer(df_max_freq=self.max_df, df_min_freq=self.min_df, vocabulary=self.vocabulary,
                                         token_pattern=r"(?u)\b\w{" + str(self.min_chars) + ",}\b",
                                         lowercase=self.lowercase, stop_words=self.stopwords)

        elif self.max_features is not None:
            self.preprocessing_steps.append('filter vocabulary to ' + str(self.max_features) + ' terms')
            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")
            vectorizer = TfidfVectorizer(df_max_freq=self.max_df, df_min_freq=self.min_df, lowercase=self.lowercase,
                                         max_features=self.max_features, stop_words=self.stopwords,
                                         token_pattern=r"(?u)\b\w{" + str(self.min_chars) + ",}\b")

        else:
            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")
            vectorizer = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df, lowercase=self.lowercase,
                                         token_pattern=r"(?u)\b\w{" + str(self.min_chars) + r",}\b",
                                         stop_words=self.stopwords)

        vectorizer.fit_transform(docs)
        vocabulary = vectorizer.get_feature_names()
        return vocabulary

    def _foo(self, docs, vocabulary, labels_path):
        final_docs, final_labels = [], []
        if labels_path is not None:
            labels = [line.strip() for line in open(labels_path, 'r').readlines()]
            for doc, label in zip(docs, labels):
                new_doc = [w for w in doc.split() if w in set(vocabulary)]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
                    final_labels.append(label)
            return final_docs, final_labels
        else:
            for doc in docs:
                new_doc = [w for w in doc.split() if w in set(vocabulary)]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
            return final_docs, []

    def simple_preprocessing_steps(self, docs):
        tmp_docs = []
        for d in docs:
            new_d = d
            if self.strip:
                new_d = new_d.replace('\n', '')
                new_d = new_d.replace('\t', '')
            if self.lowercase:
                new_d = new_d.lower()
            if self.lemmatize:
                if self.remove_stopwords_spacy:
                    new_d = ' '.join([token.lemma_ for token in self.spacy_model(new_d) if not token.is_stop])
                elif self.stopwords:
                    new_d = ' '.join([token.lemma_ for token in self.spacy_model(new_d) if token.lemma_ not in set(self.stopwords)])
                else:
                    new_d = ' '.join([token.lemma_ for token in self.spacy_model(new_d)])

            if self.remove_punctuation:
                new_d = new_d.translate(str.maketrans(self.punctuation, ' '*len(self.punctuation)))
            if self.remove_numbers:
                new_d = new_d.translate(str.maketrans("0123456789", ' '*len("0123456789")))
            new_d = " ".join(new_d.split())
            tmp_docs.append(new_d)
        return tmp_docs
