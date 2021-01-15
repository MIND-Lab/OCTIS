from octis.evaluation_metrics.metrics import Abstract_Metric
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
from gensim.models import KeyedVectors
import gensim.downloader as api
import octis.configuration.citations as citations
import octis.configuration.defaults as defaults
import numpy as np
import itertools
from scipy import spatial
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from operator import add


class Coherence(Abstract_Metric):
    def __init__(self, metric_parameters=None):
        """
        Initialize metric

        Parameters
        ----------
        metric_parameters : dictionary with keys
                            texts : list of documents (lis of lists of strings)
                            topk : how many most likely words to consider in
                            the evaluation
                            measure : (default 'c_npmi') measure to use.
                            other measures: 'u_mass', 'c_v', 'c_uci', 'c_npmi'
        """
        super().__init__()
        if metric_parameters is None:
            metric_parameters = {}
        parameters = defaults.em_coherence.copy()
        parameters.update(metric_parameters)

        self.texts = parameters['texts']
        self.dictionary = Dictionary(self.texts)
        self.topk = parameters['topk']
        self.measure = parameters['measure']
        self.parameters=parameters

    def info(self):
        return {
            "citation": citations.em_coherence,
            "name": "Coherence"
        }

    def score(self, model_output):
        """
        Retrieve the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       key 'topics' required.

        Returns
        -------
        score : coherence score
        """
        self.topics = model_output["topics"]
        if self.topics is None:
            return -1
        if self.topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            npmi = CoherenceModel(
                topics=self.topics,
                texts=self.texts,
                dictionary=self.dictionary,
                coherence=self.measure,
                processes=1,
                topn=self.topk)
            return npmi.get_coherence()


class Coherence_word_embeddings(Abstract_Metric):
    def __init__(self, metric_parameters=None):
        """
        Initialize metric

        Parameters
        ----------
        metric_parameters : dictionary with keys
                            topk : how many most likely words to consider
                            word2vec_path : if word2vec_file is specified
                            retrieves word embeddings file (in word2vec format)
                            to compute similarities, otherwise
                            'word2vec-google-news-300' is downloaded
                            binary : True if the word2vec file is binary
                            False otherwise (default False)
        """
        super().__init__()
        if metric_parameters is None:
            metric_parameters = {}
        parameters = defaults.em_coherence_we.copy()
        parameters.update(metric_parameters)
        self.parameters=parameters

        self.binary = parameters['binary']
        self.topk = parameters['topk']
        word2vec_path = parameters['word2vec_path']
        if word2vec_path is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = KeyedVectors.load_word2vec_format(
                word2vec_path, binary=self.binary)

    def info(self):
        return {
            "citation": citations.em_coherence_we,
            "name": "Coherence word embeddings"
        }

    def score(self, model_output):
        """
        Retrieve the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       key 'topics' required.

        Returns
        -------
        score : topic coherence computed on the word embeddings
                similarities
        """
        self.topics = model_output["topics"]
        if self.topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            arrays = []
            for _, topic in enumerate(self.topics):
                if len(topic) > 0:
                    local_simi = []
                    for w1, w2 in itertools.combinations(topic[0:self.topk], 2):
                        if w1 in self.wv.vocab and w2 in self.wv.vocab:
                            local_simi.append(self.wv.similarity(w1, w2))
                    arrays.append(np.mean(local_simi))
            return np.mean(arrays)


class Coherence_word_embeddings_pairwise(Abstract_Metric):
    def __init__(self, metric_parameters=None):
        """
        Initialize metric

        Parameters
        ----------
        metric_parameters : dictionary with keys
                            topk : how many most likely words to consider
                            w2v_model : a word2vector model, if not provided,
                            google news 300 will be used instead
        """
        super().__init__()
        if metric_parameters is None:
            metric_parameters = {}
        parameters = defaults.em_coherence_we_pc.copy()
        parameters.update(metric_parameters)
        self.parameters=parameters

        self.topk = parameters['topk']
        if parameters['w2v_model'] is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = parameters['w2v_model'].wv

    def info(self):
        return {
            "citation": citations.em_word_embeddings_pc,
            "name": "Coherence word embeddings pairwise"
        }

    def score(self, model_output):
        """
        Retrieve the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       key 'topics' required.

        Returns
        -------
        score : topic coherence computed on the word embeddings
        """
        self.topics = model_output["topics"]
        if self.topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            result = 0.0
            for topic in self.topics:
                E = []

                # Create matrix E (normalize word embeddings of
                #   words represented as vectors in wv)
                for word in topic[0:self.topk]:
                    if word in self.wv.vocab:
                        word_embedding = np.reshape(
                            self.wv.__getitem__(word), (1, -1))
                        normalized_we = np.reshape(
                            normalize(word_embedding), (1, -1))
                        E.append(normalized_we[0])
                E = np.array(E)

                # Perform cosine similarity between E rows
                distances = np.sum(pairwise_distances(E, metric='cosine'))
                topic_coherence = distances/(2*self.topk*(self.topk-1))

                # Update result with the computed coherence of the topic
                result += topic_coherence
            result = result/len(self.topics)
            return result


class Coherence_word_embeddings_centroid(Abstract_Metric):
    def __init__(self, metric_parameters=None):
        """
        Initialize metric

        Parameters
        ----------
        metric_parameters : dictionary with keys
                            topk : how many most likely words to consider
                            w2v_model : a word2vector model, if not provided,
                            google news 300 will be used instead
        """
        super().__init__()
        if metric_parameters is None:
            metric_parameters = {}
        parameters = defaults.em_coherence_we_pc.copy()
        parameters.update(metric_parameters)
        self.parameters=parameters

        self.topk = parameters['topk']
        if parameters['w2v_model'] is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = parameters['w2v_model'].wv

    def info(self):
        return {
            "citation": citations.em_word_embeddings_pc,
            "name": "Coherence word embeddings centroid"
        }

    def score(self, model_output):
        """
        Retrieve the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       key 'topics' required.

        Returns
        -------
        score : topic coherence computed on the word embeddings
        """
        self.topics = model_output["topics"]
        if self.topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            result = 0
            for topic in self.topics:
                E = []

                # average vector of the words in topic (centroid)
                len_word_embedding = 0
                for word in topic:
                    if word in self.wv.vocab:
                        len_word_embedding = len(self.wv.__getitem__(word))

                t = [0] * len_word_embedding

                # Create matrix E (normalize word embeddings of
                # words represented as vectors in wv) and
                # average vector of the words in topic
                for word in topic:
                    if word in self.wv.vocab:
                        word_embedding = np.reshape(
                            self.wv.__getitem__(word), (1, -1))
                        normalized_we = np.reshape(
                            normalize(word_embedding), (1, -1))
                        E.append(normalized_we[0])
                        t = list(map(add, t, word_embedding))
                t = np.array(t)
                t = t/(len(t)*sum(t))

                topic_coherence = 0
                # Perform cosine similarity between each word embedding in E
                # and t.
                for word_embedding in E:
                    distance = spatial.distance.cosine(word_embedding, t)
                    topic_coherence += distance
                topic_coherence = topic_coherence/self.topk

                # Update result with the computed coherence of the topic
                result += topic_coherence
            result /= len(self.topics)
            return result
