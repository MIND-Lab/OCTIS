from octis.evaluation_metrics.metrics import AbstractMetric
import octis.configuration.citations as citations
import itertools
import numpy as np
from octis.evaluation_metrics.rbo import rbo
from octis.evaluation_metrics.word_embeddings_rbo import word_embeddings_rbo
from octis.evaluation_metrics.word_embeddings_rbo_centroid import word_embeddings_rbo as weirbo_centroid
import gensim.downloader as api
from gensim.models import KeyedVectors


class TopicDiversity(AbstractMetric):
    def __init__(self, topk=10):
        """
        Initialize metric

        Parameters
        ----------
        topk: top k words on which the topic diversity will be computed
        """
        AbstractMetric.__init__(self)
        self.topk = topk

    def info(self):
        return {
            "citation": citations.em_topic_diversity,
            "name": "Topic diversity"
        }

    def score(self, model_output):
        """
        Retrieves the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       key 'topics' required.

        Returns
        -------
        td : score
        """
        topics = model_output["topics"]
        if topics is None:
            return 0
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than ' + str(self.topk))
        else:
            unique_words = set()
            for topic in topics:
                unique_words = unique_words.union(set(topic[:self.topk]))
            td = len(unique_words) / (self.topk * len(topics))
            return td


class InvertedRBO(AbstractMetric):
    def __init__(self, topk=10, weight=0.9):
        """
        Initialize metric Inverted Ranked-Biased Overlap

        :param topk: top k words on which the topic diversity will be computed
        :param weight: weight of each agreement at depth d. When set to 1.0, there is no weight, the rbo returns to
        average overlap. (default 0.9)
        """
        super().__init__()
        self.topk = topk
        self.weight = weight

    def score(self, model_output):
        """
        Retrieves the score of the metric

        :param model_output : dictionary, output of the model. the 'topics' key is required.

        """
        topics = model_output['topics']
        if topics is None:
            return 0
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            collect = []
            for list1, list2 in itertools.combinations(topics, 2):
                word2index = get_word2index(list1, list2)
                indexed_list1 = [word2index[word] for word in list1]
                indexed_list2 = [word2index[word] for word in list2]
                rbo_val = rbo(indexed_list1[:self.topk], indexed_list2[:self.topk], p=self.weight)[2]
                collect.append(rbo_val)
            return 1 - np.mean(collect)


class WordEmbeddingsInvertedRBO(AbstractMetric):

    def __init__(self, topk=10, weight=0.9, normalize=True, word2vec_path=None, binary=True):
        """
        Initialize metric WE-IRBO-Match

        Parameters
        ----------
        :param topk: top k words on which the topic diversity will be computed
        :param word2vec_path: word embedding space in gensim word2vec format
        :param weight: Weight of each agreement at depth d. When set to 1.0, there is no weight, the rbo returns to
        average overlap. (Default 0.9)
        :param normalize: if true, normalize the cosine similarity
        """
        super().__init__()
        self.topk = topk
        self.weight = weight
        self.norm = normalize
        self.binary = binary
        self.word2vec_path = word2vec_path
        if word2vec_path is None:
            self._wv = api.load('word2vec-google-news-300')
        else:
            self._wv = KeyedVectors.load_word2vec_format(word2vec_path, binary=self.binary)

    def score(self, model_output):
        """
        :return: rank_biased_overlap over the topics
        """
        topics = model_output['topics']
        if topics is None:
            return 0
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            collect = []
            for list1, list2 in itertools.combinations(topics, 2):
                word2index = get_word2index(list1, list2)
                index2word = {v: k for k, v in word2index.items()}
                indexed_list1 = [word2index[word] for word in list1]
                indexed_list2 = [word2index[word] for word in list2]
                rbo_val = word_embeddings_rbo(
                    indexed_list1[:self.topk], indexed_list2[:self.topk], p=self.weight,
                    index2word=index2word, word2vec=self._wv, norm=self.norm)[2]
                collect.append(rbo_val)
            return 1 - np.mean(collect)


def get_word2index(list1, list2):
    words = set(list1)
    words = words.union(set(list2))
    word2index = {w: i for i, w in enumerate(words)}
    return word2index


class WordEmbeddingsInvertedRBOCentroid(AbstractMetric):
    def __init__(self, topk=10, weight=0.9, normalize=True, word2vec_path=None, binary=True):
        super().__init__()
        self.topk = topk
        self.weight = weight
        self.norm = normalize
        self.binary = binary
        self.word2vec_path = word2vec_path
        if word2vec_path is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = KeyedVectors.load_word2vec_format( word2vec_path, binary=self.binary)

    def score(self, model_output):
        """
        :return: rank_biased_overlap over the topics
        """
        topics = model_output['topics']
        if topics is None:
            return 0
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            collect = []
            for list1, list2 in itertools.combinations(topics, 2):
                word2index = get_word2index(list1, list2)
                index2word = {v: k for k, v in word2index.items()}
                indexed_list1 = [word2index[word] for word in list1]
                indexed_list2 = [word2index[word] for word in list2]
                rbo_val = weirbo_centroid(
                    indexed_list1[:self.topk], indexed_list2[:self.topk], p=self.weight, index2word=index2word,
                    embedding_space=self.wv, norm=self.norm)[2]

                collect.append(rbo_val)
            return 1 - np.mean(collect)


class LogOddsRatio(AbstractMetric):
    def __init__(self):
        """
        Initialize metric Log Odds Ratio
        """
        super().__init__()

    def score(self, model_output):
        beta = model_output['topic-word-matrix']
        lor = 0
        count = 0
        for i, j in itertools.combinations(range(len(beta)), 2):
            lor += _LOR(beta[i], beta[j])
            count += 1
        return lor / count


def _LOR(P, Q):
    lor = 0
    for v, w in zip(P, Q):
        if v > 0 or w > 0:
            lor = lor + np.abs(np.log(v) - np.log(w))
    return lor / len(P)


class KLDivergence(AbstractMetric):
    def __init__(self):
        """
        Initialize metric Kullback-Leibler Divergence between topic-word distributions
        """
        super().__init__()

    def score(self, model_output):
        beta = model_output['topic-word-matrix']
        kl_div = 0
        count = 0
        for i, j in itertools.combinations(range(len(beta)), 2):
            kl_div += _LOR(beta[i], beta[j])
            count += 1
        return kl_div / count


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
