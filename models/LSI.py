from models.model import Abstract_Model
from gensim.models import lsimodel
import numpy as np


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

    def get_word_topic_weights(self):
        """
        Return False if the model is not trained,
        return the topic word weight matrix otherwise
        """
        if self.trained:
            topic_word_matrix = self.trained_model.get_topics()
            result = []
            for words_w in topic_word_matrix:
                minimum = min(words_w)
                words = words_w - minimum
                result.append([float(i)/sum(words) for i in words])
            return np.array(result)
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
            for word_tuple in self.trained_model.show_topic(i):
                topic_words_list.append(word_tuple[0])
            result.append(topic_words_list)
        return result

    def get_document_topic_weights(self, corpus):
        """
        Return False if the model is not trained,
        return the topic weights for the documents
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

    def get_normalized_document_topic_weights(self, corpus):
        """
        Return False if the model is not trained,
        return the normalized topic weights for the documents
        of the corpus otherwise

        Parameters
        ----------
        corpus : a corpus

        Returns
        -------
        the normalized topic weights of the documents
        of the corpus
        """
        if self.trained:
            topic_weights = self.get_document_topic_weights(corpus)
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

                # For each topic compute normalized weight
                # in the form (value-min)/(max-min)
                topic_w = []
                for topic in document_topic_weights:
                    topic_w.append((topic[0], (topic[1]-min)/(max-min)))
                result.append(topic_w)
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
            self.trained_model = lsimodel.LsiModel.load(
                model_path+"/trained_model")
