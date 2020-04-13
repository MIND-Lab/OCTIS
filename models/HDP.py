from models.model import Abstract_Model
from dataset.dataset import Dataset
import re
import numpy as np
from gensim.models import hdpmodel
import gensim.corpora as corpora


class HDP_Model(Abstract_Model):

    def set_default_hyperparameters(self):
        """
        Set hyperparameters default values for the model
        """
        self.hyperparameters = {
            'corpus': None,
            'id2word': None,
            'max_chunks': None,
            'max_time': None,
            'chunksize': 256,
            'kappa': 1.0,
            'tau': 64.0,
            'K': 15,
            'T': 150,
            'alpha': 1,
            'gamma': 1,
            'eta': 0.01,
            'scale': 1.0,
            'var_convergence': 0.0001,
            'outputdir': None,
            'random_state': None}

    def train_model(self):
        """
        Train the model and save all the data
        in trained_model
        """
        if not self.builded:
            self.build_model()

        hyperparameters = self.hyperparameters
        self.trained_model = hdpmodel.HdpModel(
            corpus=self.id_corpus,
            id2word=self.id2word,
            max_chunks=hyperparameters["max_chunks"],
            max_time=hyperparameters["max_time"],
            chunksize=hyperparameters["chunksize"],
            kappa=hyperparameters["kappa"],
            tau=hyperparameters["tau"],
            K=hyperparameters["K"],
            T=hyperparameters["T"],
            alpha=hyperparameters["alpha"],
            gamma=hyperparameters["gamma"],
            eta=hyperparameters["eta"],
            scale=hyperparameters["scale"],
            var_converge=hyperparameters["var_convergence"],
            outputdir=hyperparameters["outputdir"],
            random_state=hyperparameters["random_state"]
        )
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
            return self.trained_model[self.id2word.doc2bow(document)]
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
        for i in range(len(self.trained_model.get_topics())):
            result.append(self.trained_model.show_topic(
                i,
                topk,
                False,
                True
            ))
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
                len(self.trained_model.get_topics()),
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
            self.trained_model = hdpmodel.HdpModel.load(
                model_path+"/trained_model")
