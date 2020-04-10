from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import KeyedVectors
import gensim.downloader as api
from abc import ABC, abstractmethod
import numpy as np
import itertools


class Abstract_Metric(ABC):
    """
    Class structure of a generic metric implementation
    """

    def __init__(self):
        pass

    @abstractmethod
    def score(self):
        """
        Retrieves the score of the metric
        """
        pass


def _KL(P, Q):
    """
    Perform Kullback-Leibler divergence

    Parameters
    ----------
    P : distribution P
    Q : distribution Q

    Returns
    -------
    divergence : divergence from Q to P
    """
    # add epsilon to grant absolute continuity
    epsilon = 0.00001
    P = P+epsilon
    Q = Q+epsilon

    divergence = np.sum(P*np.log(P/Q))
    return divergence


class Topic_diversity(Abstract_Metric):
    def __init__(self, topics):
        """
        Initialize metric

        Parameters
        ----------
        topics : lists of the words of each topic
        """
        self.topics = topics

    def score(self, topk=25):
        """
        Retrieves the score of the metric

        Parameters
        ----------
        topk : top k words on which the topic diversity
               will be computed

        Returns
        -------
        td : score
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than '+str(topk))
        else:
            unique_words = set()
            for topic in self.topics:
                unique_words = unique_words.union(set(topic[:topk]))
            td = len(unique_words) / (topk * len(self.topics))
            return td


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


class KL_uniform(Abstract_Metric):
    def __init__(self, phi):
        """
        Initialize metric

        Parameters
        ----------
        phi : distribution of topics over words matrix
              phi[topic][word]
        """
        self.phi = phi

    def score(self):
        """
        Retrieves the score of the metric

        Returns
        -------
        result : score
        """
        # make uniform distribution
        val = 1.0 / len(self.phi[0])
        unif_distr = np.full(len(self.phi[0]), val)

        divergences = []
        for topic in range(len(self.phi)):

            # normalize phi, sum up to 1
            P = self.phi[topic] / self.phi[topic].sum()

            divergence = _KL(P, unif_distr)
            divergences.append(divergence)

        # KL-uniform = mean of the divergences
        # between topic-word distributions and uniform distribution
        result = np.array(divergences).mean()
        return result


class KL_vacuous(Abstract_Metric):
    def __init__(self, phi, theta):
        """
        Initialize metric

        Parameters
        ----------
        phi : distribution of topics over words matrix
              phi[topic][word]
        theta : distribution of topics over documents matrix
                theta[topic][document]
        """
        self.phi = phi
        self.theta = theta

    def score(self):
        """
        Retrieves the score of the metric

        Returns
        -------
        result : score
        """
        vacuous = np.zeros(self.phi.shape[1])
        for topic in range(len(self.theta)):

            # get probability of the topic in the corpus
            p_topic = self.theta[topic].sum()/len(self.theta[0])

            # get probability of the words:
            # P(Wi | vacuous_dist) = P(Wi | topic)*P(topic)
            vacuous += self.phi[topic]*p_topic

        divergences = []
        for topic in range(len(self.phi)):

            # normalize phi, sum up to 1
            P = self.phi[topic] / self.phi[topic].sum()

            divergence = _KL(P, vacuous)
            divergences.append(divergence)

        # KL-vacuous = mean of the divergences
        # between topic-word distributions and vacuous distribution
        result = np.array(divergences).mean()
        return result


class KL_background(Abstract_Metric):
    def __init__(self, theta):
        """
        Initialize metric

        Parameters
        ----------
        theta : distribution of topics over documents matrix
                theta[topic][document]
        """
        self.theta = theta

    def score(self):
        """
        Retrieves the score of the metric

        Returns
        -------
        result : score
        """
        # make uniform distribution
        val = 1.0 / len(self.theta[0])
        unif_distr = np.full(len(self.theta[0]), val)

        divergences = []
        for topic in range(len(self.theta)):

            # normalize theta, sum up to 1
            P = self.theta[topic] / self.theta[topic].sum()

            divergence = _KL(P, unif_distr)
            divergences.append(divergence)

        # KL-background = mean of the divergences
        # between topic-doc distributions and uniform distribution
        result = np.array(divergences).mean()
        return result
