import string
from typing import List, Union

import spacy
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from octis.dataset.dataset import Dataset

language_mapping = {'chinese': 'zh', 'danish': 'nl', 'dutch': 'nl', 'english': 'en', 'french': 'fr', 'german': 'de',
                    'greek': 'el', 'italian': 'it', 'japanese': 'ja', 'lithuanian': 'lt', 'norwegian': 'nb',
                    'polish': 'pl', 'portoguese': 'pt', 'romanian': 'ro', 'russian': 'ru', 'spanish': 'es'}


class Preprocessing:
    def __init__(self, lowercase=True, vocabulary=None, max_features=None, min_df=0, max_df=1.0,
                 remove_punctuation=True, punctuation=string.punctuation, lemmatize=True, remove_stopwords=True,
                 stopword_list: Union[str, List[str]] = 'english', min_chars=1, min_words_docs=0, language='english',
                 split=True):
        self.vocabulary = vocabulary
        self.lowercase = lowercase
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.remove_punctuation = remove_punctuation
        self.punctuation = punctuation
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.split=split
        if type(stopword_list) == list:
            stopwords = set(stopword_list)
        else:
            if 'english' in stopword_list:
                with open('octis/preprocessing/stopwords/english.txt') as fr:
                    stopwords = [line.strip() for line in fr.readlines()]
            else:
                stopwords = nltk_stopwords.words(stopword_list)  # exception if language not present
            stopwords = set(stopwords)

        self.stopwords = stopwords
        self.min_chars = min_chars
        self.min_doc_words = min_words_docs
        self.preprocessing_steps = []
        self.language = language

    def preprocess_dataset(self, documents_path, labels_path=None):
        docs = [line.strip() for line in open(documents_path, 'r').readlines()]

        if self.lowercase:
            self.preprocessing_steps.append("lowercase")
            docs = [d.lower() for d in docs]
        if self.remove_punctuation:
            self.preprocessing_steps.append('remove_punctuation')
            docs = [doc.translate(str.maketrans(self.punctuation, ' ' * len(self.punctuation))).replace(
                ' ' * 4, ' ').replace(' ' * 3, ' ').replace(' ' * 2, ' ').strip() for doc in docs]
        if self.remove_stopwords:
            self.preprocessing_steps.append('remove_stopwords')
            docs = [' '.join([w for w in doc.split() if w not in self.stopwords]) for doc in docs]
        if self.lemmatize:
            self.preprocessing_steps.append('lemmatize')
            docs = lemmatization(docs, self.language)

        vocabulary = self.filter_other_words(docs)

        final_docs, final_labels = [], []
        if labels_path is not None:
            labels = [line.strip() for line in open(labels_path, 'r').readlines()]
            for doc, label in zip(docs, labels):
                new_doc = [w for w in doc.split() if w in set(vocabulary)]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
                    final_labels.append(label)
        else:
            labels = []
            for doc in docs:
                new_doc = [w for w in doc.split() if w in set(vocabulary)]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
        self.preprocessing_steps.append('filter documents with less than ' + str(self.min_doc_words) + " words")

        metadata = {"total_documents": len(docs), "vocabulary_length": len(vocabulary),
                    "preprocessing-info": self.preprocessing_steps, "labels": list(set(labels)),
                    "total_labels": len(labels)}
        if self.split:
            if len(labels)>0:
                train, test, y_train, y_test = train_test_split(
                    range(len(final_docs)), final_labels, test_size=0.15, random_state=1, stratify=final_labels)

                train, validation = train_test_split(train, test_size=3 / 17, random_state=1, stratify=y_train)
                partitioned_labels = [final_labels[doc] for doc in train + validation + test]
                partitioned_corpus = [final_docs[doc] for doc in train + validation + test]
                metadata["last-training-doc"] = len(train)
                metadata["last-validation-doc"] = len(validation) + len(train)

                return Dataset(partitioned_corpus, vocabulary, metadata, partitioned_labels)
            else:

                train, test = train_test_split(range(len(final_docs)), test_size=0.15, random_state=1)
                train, validation = train_test_split(train, test_size=3 / 17, random_state=1)

                metadata["last-training-doc"] = len(train)
                metadata["last-validation-doc"] = len(validation) + len(train)
                partitioned_corpus = [final_docs[doc] for doc in train + validation + test]

                return Dataset(partitioned_corpus, vocabulary, metadata, labels)
        else:
            Dataset(final_docs, vocabulary, metadata, labels)

    def filter_other_words(self, docs):
        if self.vocabulary is not None:
            self.preprocessing_steps.append('filter words by vocabulary')
            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")

            vectorizer = TfidfVectorizer(df_max_freq=self.max_df, df_min_freq=self.min_df, vocabulary=self.vocabulary,
                                         token_pattern=r"(?u)\b\w{" + str(self.min_chars) + ",}\b",
                                         lowercase=self.lowercase)
        elif self.max_features is not None:
            self.preprocessing_steps.append('filter vocabulary to ' + str(self.max_features) + ' terms')
            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")

            vectorizer = TfidfVectorizer(df_max_freq=self.max_df, df_min_freq=self.min_df, lowercase=self.lowercase,
                                         max_features=self.max_features,
                                         token_pattern=r"(?u)\b\w{" + str(self.min_chars) + ",}\b")
        else:
            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")
            vectorizer = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df, lowercase=self.lowercase,
                                         token_pattern=r"(?u)\b\w{" + str(self.min_chars) + r",}\b")
        vectorizer.fit_transform(docs)
        vocabulary = vectorizer.get_feature_names()
        return vocabulary


def lemmatization(corpus, language):
    """
    Lemmatize the words in the corpus

    Parameters
    ----------
    corpus: the corpus
    Returns
    -------
    result : corpus lemmatized
    """
    lang = language_mapping[language]
    try:
        nlp = spacy.load(lang + "_core_web_sm")
    except IOError:
        raise IOError("Can't find model " + lang +
                      "_core_web_sm. Check the data directory or download it using the following command:"
                      "\npython -m spacy download " + lang + "_core_web_sm")
    return [' '.join([token.lemma_ for token in nlp(document)]) for document in corpus]


