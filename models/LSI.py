from models.model import Abstract_Model
from dataset.dataset import Dataset
import re
from gensim.models import lsimodel
import gensim.corpora as corpora


class LSI_Model(Abstract_Model):

    def set_default_hyperparameters(self):
        """
        Set hyperparameters default values for the model
        """
        self.hyperparameters = {
            'corpus': None,
            'num_topics': 100,
            'id2word': None,
            'distributed': False,
            'chunksize': 20000,
            'decay': 1.0,
            'onepass': True,
            'power_iters': 2,
            'extra_samples': 100}

    def map_vocabulary(self):
        """
        Create a dictionary to allow fast retrieving
        of a word from an Id.
        Id's are used to represent the words of
        the vocabulary
        """
        self.id2word = corpora.Dictionary(self.dataset.get_corpus())

    def build_model(self):
        """
        Adapt the corpus to the model
        """
        self.id_corpus = [self.id2word.doc2bow(
            document) for document in self.dataset.get_corpus()]
        self.builded = True
        self.trained = False

    def train_model(self):
        """
        Train the model and save all the data
        in trained_model
        """
        if not self.builded:
            self.build_model()

        hyperparameters = self.hyperparameters
        self.trained_model = lsimodel.LsiModel(
            corpus=self.id_corpus,
            id2word=self.id2word,
            num_topics=hyperparameters["num_topics"],
            distributed=hyperparameters["distributed"],
            chunksize=hyperparameters["chunksize"],
            decay=hyperparameters["decay"],
            onepass=hyperparameters["onepass"],
            power_iters=hyperparameters["power_iters"],
            extra_samples=hyperparameters["extra_samples"])
        self.trained = True
        return True

    def make_topic_word_matrix(self):
        """
        Return False if the model is not trained,
        produce the topic word matrix
        and return True otherwise
        """
        if self.trained:
            self.topic_word_matrix = self.trained_model.get_topics()
            return True
        return False

    def get_doc_topic_weights(self, corpus):
        """
        Return False if the model is not trained,
        return the topic weigths for the document
        of the corpus otherwise

        Parameters
        ----------
        corpus : a corpus

        Returns
        -------
        the topic weights of the documents
        of the corpus
        """
        if self.trained:
            corpus = [self.id2word.doc2bow(document) for document in corpus]
            return self.trained_model[corpus]
        return False

    def get_normalized_topic_weigths(self, corpus):
        """
        Return False if the model is not trained,
        return the topic weigths for the document
        of the corpus otherwise

        Parameters
        ----------
        corpus : a corpus

        Returns
        -------
        the topic weights of the documents
        of the corpus
        """
        if self.trained:
            topic_weights = self.get_doc_topic_weights(corpus)
            result = []

            for document_topic_weights in topic_weights:

                # Find min e max topic_weights values
                min = document_topic_weights[0][1]
                max = document_topic_weights[0][1]
                for topic in document_topic_weights:
                    if topic[1] > max:
                        max = topic[1]
                    if topic[1] < min:
                        min = topic[1]

                # For each topic compute normalized weitght
                # in the form (value-min)/(max-min)
                topic_w = []
                for topic in document_topic_weights:
                    topic_w.append((topic[0], (topic[1]-min)/(max-min)))
                result.append(topic_w)
            return result
        return False
