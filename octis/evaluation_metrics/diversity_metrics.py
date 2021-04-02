from octis.evaluation_metrics.metrics import AbstractMetric
import octis.configuration.citations as citations
import octis.configuration.defaults as defaults
import itertools
import numpy as np
from octis.evaluation_metrics.rbo import rbo
from octis.evaluation_metrics.word_embeddings_rbo import word_embeddings_rbo
from octis.evaluation_metrics.word_embeddings_rbo_centroid import word_embeddings_rbo as weirbo_centroid


class TopicDiversity(AbstractMetric):
    def __init__(self, metric_parameters=None):
        """
        Initialize metric

        Parameters
        ----------
        metric_parameters : dictionary with key 'topk'
                            topk: top k words on which the topic diversity
                            will be computed
        """
        AbstractMetric.__init__(self, metric_parameters)
        if metric_parameters is None:
            metric_parameters = {}
        parameters = defaults.em_topic_diversity.copy()
        parameters.update(metric_parameters)
        self.topk = parameters["topk"]
        self.parameters = parameters

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
    def __init__(self, metric_parameters=None):
        """
        Initialize metric

        Parameters
        ----------
        metric_parameters : dictionary with keys 'topk' and 'weight'
                            topk: top k words on which the topic diversity
                                  will be computed
                            weight: p (float), default 1.0: Weight of each
                                    agreement at depth d:p**(d-1). When set
                                    to 1.0, there is no weight, the rbo returns
                                    to average overlap.
        """
        super().__init__(metric_parameters)
        if metric_parameters is None:
            metric_parameters = {}
        parameters = defaults.em_invertedRBO.copy()
        parameters.update(metric_parameters)
        self.parameters = parameters

        self.topk = parameters["topk"]
        self.weight = parameters["weight"]
        self.topics = None

    def score(self, model_output):
        """
        Retrieves the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       key 'topics' required.

        Returns
        -------
        td : score of the rank biased overlap over tht topics
        """
        self.topics = model_output['topics']
        if self.topics is None or type(self.topics) != list:
            return 0
        if self.topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            collect = []
            for list1, list2 in itertools.combinations(self.topics, 2):
                word2index = self.get_word2index(list1, list2)
                indexed_list1 = [word2index[word] for word in list1]
                indexed_list2 = [word2index[word] for word in list2]
                rbo_val = rbo(
                    indexed_list1[:self.topk],
                    indexed_list2[:self.topk],
                    p=self.weight)[2]
                collect.append(rbo_val)
            return 1 - np.mean(collect)

    def get_word2index(self, list1, list2):
        words = set(list1)
        words = words.union(set(list2))
        word2index = {w: i for i, w in enumerate(words)}
        return word2index


class WordEmbeddingsInvertedRBO(AbstractMetric):
    def __init__(self, metric_parameters=None):
        super().__init__(metric_parameters)
        if metric_parameters is None:
            metric_parameters = {}
        parameters = defaults.em_word_embeddings_invertedRBO.copy()
        parameters.update(metric_parameters)
        self.parameters = parameters
        self.topk = metric_parameters["topk"]
        self.weight = metric_parameters["weight"]
        self.norm = metric_parameters["norm"]
        self.word_embedding_model = metric_parameters['embedding_model']

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
                word2index = self.get_word2index(list1, list2)
                index2word = {v: k for k, v in word2index.items()}
                indexed_list1 = [word2index[word] for word in list1]
                indexed_list2 = [word2index[word] for word in list2]
                rbo_val = word_embeddings_rbo(indexed_list1[:self.topk],
                                              indexed_list2[:self.topk], p=self.weight,
                                              index2word=index2word,
                                              word2vec=self.word_embedding_model,
                                              norm=self.norm)[2]
                collect.append(rbo_val)
            return 1 - np.mean(collect)

    def get_word2index(self, list1, list2):
        words = set(list1)
        words = words.union(set(list2))
        word2index = {w: i for i, w in enumerate(words)}
        return word2index


class WordEmbeddingsInvertedRBOCentroid(AbstractMetric):
    def __init__(self, metric_parameters=None):
        super().__init__(metric_parameters)
        if metric_parameters is None:
            metric_parameters = {}
        parameters = defaults.em_word_embeddings_invertedRBO.copy()
        parameters.update(metric_parameters)
        self.parameters = parameters
        self.topk = metric_parameters["topk"]
        self.weight = metric_parameters["weight"]
        self.norm = metric_parameters["norm"]
        self.word_embedding_model = metric_parameters['embedding_model']

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
                word2index = self.get_word2index(list1, list2)
                index2word = {v: k for k, v in word2index.items()}
                indexed_list1 = [word2index[word] for word in list1]
                indexed_list2 = [word2index[word] for word in list2]
                rbo_val = weirbo_centroid(indexed_list1[:self.topk],
                                          indexed_list2[:self.topk], p=self.weight,
                                          index2word=index2word,
                                          word2vec=self.word_embedding_model, norm=self.norm)[2]
                collect.append(rbo_val)
            return 1 - np.mean(collect)

    def get_word2index(self, list1, list2):
        words = set(list1)
        words = words.union(set(list2))
        word2index = {w: i for i, w in enumerate(words)}
        return word2index
