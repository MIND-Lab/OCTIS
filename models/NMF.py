from models.interface import Abstract_Model
from dataset.dataset import Dataset
import re
from gensim.models import nmf
import gensim.corpora as corpora


class NMF_Model(Abstract_Model):

    def set_default_hyperparameters(self):
        """
        Set hyperparameters default values for the model
        """
        self.hyperparameters = {
            'corpus': None,
            'num_topics': 100,
            'id2word': None,
            'chunksize': 2000,
            'passes': 1,
            'kappa': 1.0,
            'minimum_probability': 0.01,
            'w_max_iter': 200,
            'w_stop_condition': 0.0001,
            'h_max_iter': 50,
            'h_stop_condition': 0.001,
            'eval_every': 10,
            'normalize': True,
            'random_state': None}

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
        self.trained_model = nmf.Nmf(
            corpus=self.id_corpus,
            id2word=self.id2word,
            num_topics=hyperparameters["num_topics"],
            chunksize=hyperparameters["chunksize"],
            passes=hyperparameters["passes"],
            kappa=hyperparameters["kappa"],
            minimum_probability=hyperparameters["minimum_probability"],
            w_max_iter=hyperparameters["w_max_iter"],
            w_stop_condition=hyperparameters["w_stop_condition"],
            h_max_iter=hyperparameters["h_max_iter"],
            h_stop_condition=hyperparameters["h_stop_condition"],
            eval_every=hyperparameters["eval_every"],
            normalize=hyperparameters["normalize"],
            random_state=hyperparameters["random_state"])
        self.trained = True
        return True

    def make_topic_word_matrix(self):
        """
        Return False if the model is not trained,
        produce the document topic representation
        and return True otherwise
        """
        if self.trained:
            self.topic_word_matrix = self.trained_model.get_topics()
            return True
        return False

    def get_document_topics(self, document):
        """
        Return False if the model is not trained,
        return the topic word matrix otherwise

        Parameters
        ----------
        document : a document in format
                   list of strings (words)

        Returns
        -------
        the topic representation of the document
        """
        if self.trained:
            return self.trained_model.get_document_topics(
                self.id2word.doc2bow(document))
        return False

    def get_doc_topic_representation(self, corpus):
        """
        Return False if the model is not trained,
        return the topic word matrix otherwise

        Parameters
        ----------
        corpus : a corpus

        Returns
        -------
        the topic representation of the documents
        of the corpus
        """
        if self.trained:
            result = []
            for document in corpus:
                result.append(self.get_document_topics(document))
            return result
        return False
