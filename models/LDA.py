from models.model import Abstract_Model
from dataset.dataset import Dataset
import re
import numpy as np
from gensim.models import ldamodel
import gensim.corpora as corpora


class LDA_Model(Abstract_Model):

    def set_default_hyperparameters(self):
        """
        Set hyperparameters default values for the model
        """
        self.hyperparameters = {
            'corpus': None,
            'num_topics': 100,
            'id2word': None,
            'distributed': False,
            'chunksize': 2000,
            'passes': 1,
            'update_every': 1,
            'alpha': 'symmetric',
            'eta': None,
            'decay': 0.5,
            'offset': 1.0,
            'eval_every': 10,
            'iterations': 50,
            'gamma_threshold': 0.001,
            'minimum_probability': 0.01,
            'random_state': None,
            'ns_conf': None,
            'minimum_phi_value': 0.01,
            'per_word_topics': False,
            'callbacks': None}

    def train_model(self):
        """
        Train the model and save all the data
        in trained_model
        """
        if not self.builded:
            self.build_model()

        hyperparameters = self.hyperparameters
        self.trained_model = ldamodel.LdaModel(
            corpus=self.id_corpus,
            id2word=self.id2word,
            num_topics=hyperparameters["num_topics"],
            distributed=hyperparameters["distributed"],
            chunksize=hyperparameters["chunksize"],
            passes=hyperparameters["passes"],
            update_every=hyperparameters["update_every"],
            alpha=hyperparameters["alpha"],
            eta=hyperparameters["eta"],
            decay=hyperparameters["decay"],
            offset=hyperparameters["offset"],
            eval_every=hyperparameters["eval_every"],
            iterations=hyperparameters["iterations"],
            gamma_threshold=hyperparameters["gamma_threshold"],
            minimum_probability=hyperparameters["minimum_probability"],
            random_state=hyperparameters["random_state"],
            ns_conf=hyperparameters["ns_conf"],
            minimum_phi_value=hyperparameters["minimum_phi_value"],
            per_word_topics=hyperparameters["per_word_topics"],
            callbacks=hyperparameters["callbacks"])
        self.trained = True
        return True

    def get_word_topic_weights(self):
        """
        Return False if the model is not trained,
        return the word topic weights matrix otherwise
        """
        if self.trained:
            return self.trained_model.get_topics()
        return None

    def get_document_topics(self, document):
        """
        Return False if the model is not trained,
        return the topic representation of the
        document otherwise

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

    def get_topics_terms(self, topk=10):
        """
        Return False if the model is not trained,
        return the topk words foreach topic otherwise

        Parameters
        ----------
        topk: top k words to retrieve from each topic
              (ordered by weight)

        Returns
        -------
        result : list of lists, each list
                 contains topk words for the topic
        """
        result = []
        for i in range(self.hyperparameters["num_topics"]):
            topic_words_list = []
            for word_tuple in self.trained_model.get_topic_terms(i):
                topic_words_list.append(self.id2word[word_tuple[0]])
            result.append(topic_words_list)
        return result

    def get_doc_topic_representation(self, corpus):
        """
        Return False if the model is not trained,
        return the topic representation of the
        corpus otherwise

        Parameters
        ----------
        corpus : a corpus

        Returns
        -------
        the topic representation of the documents
        of the corpus
        """
        if self.trained:
            doc_topic_tuples = []
            for document in corpus:
                doc_topic_tuples.append(self.get_document_topics(document))

            result = np.zeros((
                self.hyperparameters["num_topics"],
                len(doc_topic_tuples)))

            for ndoc in range(len(doc_topic_tuples)):
                document = doc_topic_tuples[ndoc]
                for topic_tuple in document:
                    result[topic_tuple[0]][ndoc] = topic_tuple[1]
            return result
        return False

    def save(self, model_path, dataset_path=None):
        """
        Save the model in a folder.
        By default the dataset is not saved

        Parameters
        ----------
        model_path : model directory path
        dataset_path : dataset path (optional)
        """
        super().save(model_path, dataset_path)
        if self.trained:
            self.trained_model.save(model_path+"/trained_model")

    def load(self, model_path, dataset_path=None):
        """
        Load the model from a folder.
        By default the dataset is not loaded

        Parameters
        ----------
        model_path : model directory path
        dataset_path : dataset path (optional)
        """
        super().load(model_path, dataset_path)
        if self.trained:
            self.trained_model = ldamodel.LdaModel.load(
                model_path+"/trained_model")
