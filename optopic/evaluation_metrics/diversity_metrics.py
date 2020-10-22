from optopic.evaluation_metrics.metrics import Abstract_Metric
import optopic.configuration.citations as citations
import optopic.configuration.defaults as defaults
import itertools
import numpy as np
from optopic.evaluation_metrics.rbo import rbo
from optopic.evaluation_metrics.word_embeddings_rbo import word_embeddings_rbo


class Topic_diversity(Abstract_Metric):
    def __init__(self, metric_parameters={}):
        """
        Initialize metric

        Parameters
        ----------
        metric_parameters : dictionary with key 'topk'
                            topk: top k words on which the topic diversity
                            will be computed
        """
        Abstract_Metric.__init__(self, metric_parameters)
        parameters = defaults.em_topic_diversity.copy()
        parameters.update(metric_parameters)
        self.topk = parameters["topk"]

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
        self.topics = model_output["topics"]
        if self.topk > len(self.topics[0]):
            raise Exception('Words in topics are less than '+str(self.topk))
        else:
            unique_words = set()
            for topic in self.topics:
                unique_words = unique_words.union(set(topic[:self.topk]))
            td = len(unique_words) / (self.topk * len(self.topics))
            return td


class InvertedRBO(Abstract_Metric):
    def __init__(self, metric_parameters={}):
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
        super().__init__()
        parameters = defaults.em_invertedRBO.copy()
        parameters.update(metric_parameters)

        self. topk = parameters["topk"]
        self.weight = parameters["weight"]

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



class WordEmbeddingsInvertedRBO(Abstract_Metric):
    def __init__(self,metric_parameters=defaults.em_word_embeddings_invertedRBO.copy()):
        super().__init__()
        self.topk = metric_parameters["topk"]
        self.weight = metric_parameters["weight"]
        self.word_embedding_model = metric_parameters['embedding_model']

    def score(self, model_output):
        '''
        :param weight: p (float), default 1.0: Weight of each agreement at depth d:
        p**(d-1). When set to 1.0, there is no weight, the rbo returns to average overlap.
        :param topic_list: a list of lists of words
        :return: rank_biased_overlap over the topics
        '''
        self.topics = model_output['topics']
        if self.topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            collect = []
            for list1, list2 in itertools.combinations(self.topics, 2):
                word2index = self.get_word2index(list1, list2)
                index2word = {v: k for k, v in word2index.items()}
                indexed_list1 = [word2index[word] for word in list1]
                indexed_list2 = [word2index[word] for word in list2]
                rbo_val = word_embeddings_rbo(indexed_list1[:self.topk],
                                              indexed_list2[:self.topk], p=self.weight,
                                              index2word=index2word,
                                              word2vec=self.word_embedding_model)[2]
                collect.append(rbo_val)
            return 1 - np.mean(collect)

    def get_word2index(self, list1, list2):
        words = set(list1)
        words = words.union(set(list2))
        word2index = {w: i for i, w in enumerate(words)}
        return word2index
