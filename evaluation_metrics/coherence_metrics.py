from evaluation_metrics.metrics import Abstract_Metric
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np
import itertools
from scipy import spatial
from sklearn.metrics import pairwise_distances
from operator import add
from pathlib import Path


def _load_default_texts():
    """
    Loads default wikipedia texts

    Returns
    -------
    result : default wikipedia texts
    """
    file_name = "preprocessed_datasets/wikipedia/wikipedia_not_lemmatized_5/corpus.txt"
    result = []
    file = Path(file_name)
    if file.is_file():
        with open(file_name, 'r') as corpus_file:
            for line in corpus_file:
                result.append(line.split())
        return result
    return False


coherence_defaults = {
    'texts': _load_default_texts(),
    'topk': 10,
    'measure': 'c_npmi'
}

coherence_we_defaults = {
    'topk': 10,
    'word2vec_path': None,
    'binary': False
}

coherence_we_pc_defaults = {
    'topk': 10,
    'w2v_model': None
}

we_pc_citation = r"""
@inproceedings{DBLP:conf/emnlp/DingNX18,
  author    = {Ran Ding and
               Ramesh Nallapati and
               Bing Xiang},
  editor    = {Ellen Riloff and
               David Chiang and
               Julia Hockenmaier and
               Jun'ichi Tsujii},
  title     = {Coherence-Aware Neural Topic Modeling},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural
               Language Processing, Brussels, Belgium, October 31 - November 4, 2018},
  pages     = {830--836},
  publisher = {Association for Computational Linguistics},
  year      = {2018},
  url       = {https://doi.org/10.18653/v1/d18-1096},
  doi       = {10.18653/v1/d18-1096},
  timestamp = {Tue, 28 Jan 2020 10:28:21 +0100},
  biburl    = {https://dblp.org/rec/conf/emnlp/DingNX18.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""
coherence_citation = r"""
@inproceedings{DBLP:conf/wsdm/RoderBH15,
  author    = {Michael R{\"{o}}der and
               Andreas Both and
               Alexander Hinneburg},
  editor    = {Xueqi Cheng and
               Hang Li and
               Evgeniy Gabrilovich and
               Jie Tang},
  title     = {Exploring the Space of Topic Coherence Measures},
  booktitle = {Proceedings of the Eighth {ACM} International Conference on Web Search
               and Data Mining, {WSDM} 2015, Shanghai, China, February 2-6, 2015},
  pages     = {399--408},
  publisher = {{ACM}},
  year      = {2015},
  url       = {https://doi.org/10.1145/2684822.2685324},
  doi       = {10.1145/2684822.2685324},
  timestamp = {Tue, 21 May 2019 11:38:33 +0200},
  biburl    = {https://dblp.org/rec/conf/wsdm/RoderBH15.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""


class Coherence(Abstract_Metric):
    def __init__(self, metric_parameters=coherence_defaults):
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
        self.texts = metric_parameters['texts']
        self.dictionary = Dictionary(self.texts)
        self.topk = metric_parameters['topk']
        self.measure = metric_parameters['measure']

    def info(self):
        return {
            "citation": coherence_citation,
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
        if self.topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            npmi = CoherenceModel(
                topics=self.topics,
                texts=self.texts,
                dictionary=self.dictionary,
                coherence=self.measure,
                topn=self.topk)
            return npmi.get_coherence()


class Coherence_word_embeddings(Abstract_Metric):
    def __init__(self, metric_parameters=coherence_we_defaults):
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
        self.binary = metric_parameters['binary']
        self.topk = metric_parameters['topk']
        word2vec_path = metric_parameters['word2vec_path']
        if word2vec_path is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = KeyedVectors.load_word2vec_format(
                word2vec_path, binary=self.binary)

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
    def __init__(self, metric_parameters=coherence_we_pc_defaults):
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
        self.topk = metric_parameters['topk']
        if metric_parameters['w2v_model'] is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = metric_parameters['w2v_model'].wv

    def info(self):
        return {
            "citation": we_pc_citation,
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
                        word_embedding = self.wv.__getitem__(word)
                        normalized_we = word_embedding/word_embedding.sum()
                        E.append(normalized_we)
                E = np.array(E)

                # Perform cosine similarity between E rows
                distances = np.sum(pairwise_distances(E, metric='cosine'))
                topic_coherence = (distances)/(2*self.topk*(self.topk-1))

                # Update result with the computed coherence of the topic
                result += topic_coherence
            result = result/len(self.topics)
            return result


class Coherence_word_embeddings_centroid(Abstract_Metric):
    def __init__(self, metric_parameters=coherence_we_pc_defaults):
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
        self.topk = metric_parameters['topk']
        if metric_parameters['w2v_model'] is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = metric_parameters['w2v_model'].wv

    def info(self):
        return {
            "citation": we_pc_citation,
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
                topic_coherence = topic_coherence/self.topk

                # Update result with the computed coherence of the topic
                result += topic_coherence
            result /= len(self.topics)
            return result
