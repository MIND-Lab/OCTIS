from octis.evaluation_metrics.metrics import AbstractMetric
from octis.dataset.dataset import Dataset
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
from gensim.models import KeyedVectors
import gensim.downloader as api
import octis.configuration.citations as citations
import numpy as np
import itertools
from scipy import spatial
from sklearn.preprocessing import normalize
from operator import add


class Coherence(AbstractMetric):
    def __init__(self, texts=None, topk=10, measure='c_npmi'):
        """
        Initialize metric

        Parameters
        ----------
        texts : list of documents (list of lists of strings)
        topk : how many most likely words to consider in
        the evaluation
        measure : (default 'c_npmi') measure to use.
        other measures: 'u_mass', 'c_v', 'c_uci', 'c_npmi'
        """
        super().__init__()
        if texts is None:
            self._texts = _load_default_texts()
        else:
            self._texts = texts
        self._dictionary = Dictionary(self._texts)
        self.topk = topk
        self.measure = measure

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
        topics = model_output["topics"]
        if topics is None:
            return -1
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            npmi = CoherenceModel(topics=topics, texts=self._texts, dictionary=self._dictionary,
                                  coherence=self.measure, processes=1, topn=self.topk)
            return npmi.get_coherence()


class WECoherencePairwise(AbstractMetric):
    def __init__(self, word2vec_path=None, binary=False, topk=10):
        """
        Initialize metric

        Parameters
        ----------
        dictionary with keys
        topk : how many most likely words to consider
        word2vec_path : if word2vec_file is specified retrieves word embeddings file (in word2vec format)
        to compute similarities, otherwise 'word2vec-google-news-300' is downloaded
        binary : True if the word2vec file is binary, False otherwise (default False)
        """
        super().__init__()

        self.binary = binary
        self.topk = topk
        self.word2vec_path = word2vec_path
        if word2vec_path is None:
            self._wv = api.load('word2vec-google-news-300')
        else:
            self._wv = KeyedVectors.load_word2vec_format(
                word2vec_path, binary=self.binary)

    def info(self):
        return {
            "citation": citations.em_coherence_we,
            "name": "Coherence word embeddings pairwise cosine"
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
        topics = model_output["topics"]
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            arrays = []
            for _, topic in enumerate(topics):
                if len(topic) > 0:
                    local_simi = []
                    for w1, w2 in itertools.combinations(topic[0:self.topk], 2):
                        if w1 in self._wv.vocab and w2 in self._wv.vocab:
                            local_simi.append(self._wv.similarity(w1, w2))
                    arrays.append(np.mean(local_simi))
            return np.mean(arrays)


class WECoherenceCentroid(AbstractMetric):
    def __init__(self, topk=10, word2vec_path=None, binary=True):
        """
        Initialize metric

        Parameters
        ----------
        topk : how many most likely words to consider
        w2v_model_path : a word2vector model path, if not provided, google news 300 will be used instead
        """
        super().__init__()

        self.topk = topk
        self.binary = binary
        self.word2vec_path = word2vec_path
        if self.word2vec_path is None:
            self._wv = api.load('word2vec-google-news-300')
        else:
            self._wv = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=self.binary)

    @staticmethod
    def info():
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
        topics = model_output["topics"]
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            result = 0
            for topic in topics:
                E = []

                # average vector of the words in topic (centroid)
                len_word_embedding = 0
                for word in topic:
                    if word in self._wv.vocab:
                        len_word_embedding = len(self._wv.__getitem__(word))

                t = [0] * len_word_embedding

                # Create matrix E (normalize word embeddings of words represented as vectors in wv) and
                # average vector of the words in topic
                for word in topic:
                    if word in self._wv.vocab:
                        word_embedding = np.reshape(
                            self._wv.__getitem__(word), (1, -1))
                        normalized_we = np.reshape(
                            normalize(word_embedding), (1, -1))
                        E.append(normalized_we[0])
                        t = list(map(add, t, word_embedding))
                t = np.array(t)
                t = t/(len(t)*sum(t))

                topic_coherence = 0
                # Perform cosine similarity between each word embedding in E and t.
                for word_embedding in E:
                    distance = spatial.distance.cosine(word_embedding, t)
                    topic_coherence += distance
                topic_coherence = topic_coherence/self.topk

                # Update result with the computed coherence of the topic
                result += topic_coherence
            result /= len(topics)
            return result


def _load_default_texts():
    """
    Loads default general texts

    Returns
    -------
    result : default 20newsgroup texts
    """
    dataset = Dataset()
    dataset.fetch_dataset("20NewsGroup")
    return dataset.get_corpus()
