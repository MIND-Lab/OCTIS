from evaluation_metrics.metrics import Abstract_Metric
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np
import itertools
from scipy import spatial
from sklearn.metrics import pairwise_distances
from operator import add


class Coherence(Abstract_Metric):
    def __init__(self, topics, texts):
        """
        Initialize metric

        Parameters
        ----------
        topics : lists of the words of each topic
        texts : list of documents (lis of lists of strings)
        """
        super().__init__()
        self.topics = topics
        self.texts = texts
        self.dictionary = Dictionary(self.texts)

    def score(self, topk=10, measure='c_npmi'):
        """
        Retrieve the score of the metric

        Parameters
        ----------
        topics : a list of lists of the top-k words
        texts : (list of lists of strings) represents the corpus
                on which the empirical frequencies of words are computed
        topk : how many most likely words to consider in
               the evaluation
        measure : (default 'c_npmi') coherence measure to be used.
                  other measures: 'u_mass', 'c_v', 'c_uci', 'c_npmi'
        Returns
        -------
        score : coherence score
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            npmi = CoherenceModel(
                topics=self.topics,
                texts=self.texts,
                dictionary=self.dictionary,
                coherence=measure,
                topn=topk)
            return npmi.get_coherence()


class Coherence_word_embeddings(Abstract_Metric):
    def __init__(self, topics, word2vec_path=None, binary=False):
        """
        Initialize metric

        Parameters
        ----------
        topics : a list of lists of the top-n most likely words
        word2vec_path : if word2vec_file is specified, it retrieves
                        the word embeddings file (in word2vec format)
                        to compute similarities between words, otherwise
                        'word2vec-google-news-300' is downloaded
        binary : True if the word2vec file is binary, False otherwise
                 (default False)
        """
        super().__init__()
        self.topics = topics
        self.binary = binary
        if word2vec_path is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = KeyedVectors.load_word2vec_format(
                word2vec_path, binary=binary)

    def score(self, topk=10):
        """
        Retrieve the score of the metric

        Parameters
        ----------
        topk : how many most likely words to consider in the
               evaluation

        Returns
        -------
        score : topic coherence computed on the word embeddings
                similarities
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            arrays = []
            for _, topic in enumerate(self.topics):
                if len(topic) > 0:
                    local_simi = []
                    for w1, w2 in itertools.combinations(topic[0:topk], 2):
                        if w1 in self.wv.vocab and w2 in self.wv.vocab:
                            local_simi.append(self.wv.similarity(w1, w2))
                    arrays.append(np.mean(local_simi))
            return np.mean(arrays)

class Coherence_word_embeddings_pairwise(Abstract_Metric):
    def __init__(self, topics, w2v_model=None):
        """
        Initialize metric

        Parameters
        ----------
        topics : a list of lists of the top-n most likely words
        w2v_model : a word2vector model, if is not provided,
                    google news 300 will be used instead
        """
        super().__init__()
        self.topics = topics
        if w2v_model is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = w2v_model.wv

    def score(self, topk=10):
        """
        Retrieve the score of the metric

        Parameters
        ----------
        topk : how many most likely words to consider in the
               evaluation

        Returns
        -------
        score : topic coherence computed on the word embeddings
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            result = 0.0
            for topic in self.topics:
                E = []

                # Create matrix E (normalize word embeddings of
                #   words represented as vectors in wv)
                for word in topic[0:topk]:
                    if word in self.wv.vocab:
                        word_embedding = self.wv.__getitem__(word)
                        normalized_we = word_embedding/word_embedding.sum()
                        E.append(normalized_we)
                E = np.array(E)

                # Perform cosine similarity between E rows
                distances = np.sum(pairwise_distances(E, metric='cosine'))
                topic_coherence = (distances)/(2*topk*(topk-1))

                # Update result with the computed coherence of the topic
                result += topic_coherence
            result = result/len(self.topics)
            return result


class Coherence_word_embeddings_centroid(Abstract_Metric):
    def __init__(self, topics, w2v_model=None):
        """
        Initialize metric

        Parameters
        ----------
        topics : a list of lists of the top-n most likely words
        w2v_model : a word2vector model, if is not provided,
                    google news 300 will be used instead
        """
        super().__init__()
        self.topics = topics
        if w2v_model is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = w2v_model.wv

    def score(self, topk=10):
        """
        Retrieve the score of the metric

        Parameters
        ----------
        topk : how many most likely words to consider in the
               evaluation

        Returns
        -------
        score : topic coherence computed on the word embeddings
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            result = 0
            for topic in self.topics:
                E = []
                # average vector of the words in topic (centroid)
                t = [0] * len(self.wv.__getitem__(topic[0]))

                # Create matrix E (normalize word embeddings of
                # words represented as vectors in wv) and
                # average vector of the words in topic
                for word in topic:
                    if word in self.wv.vocab:
                        word_embedding = self.wv.__getitem__(word)
                        normalized_we = word_embedding/sum(word_embedding)
                        E.append(normalized_we)
                        t = list(map(add, t, word_embedding))
                t = np.array(t)
                t = t/(len(t)*sum(t))

                topic_coherence = 0
                # Perform cosine similarity between each word embedding in E
                # and t.
                for word_embedding in E:
                    distance = spatial.distance.cosine(word_embedding, t)
                    topic_coherence += distance
                topic_coherence = topic_coherence/topk

                # Update result with the computed coherence of the topic
                result += topic_coherence
            result /= len(self.topics)
            return result
