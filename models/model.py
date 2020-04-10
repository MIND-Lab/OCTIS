from abc import ABC, abstractmethod
from dataset.dataset import Dataset
import gensim.corpora as corpora
import json
from pathlib import Path


class Abstract_Model(ABC):
    """
    Class structure of a generic Topic Modelling implementation
    """
    # True if the model is builded
    # False otherwise
    builded = False

    # True if the model is trained
    # False otherwise
    trained = False

    # Dicitonary
    # key = id of the word
    # value = word
    id2word = {}

    # Dictionary
    # key = hyperparameter name
    # value = hyperparameter value
    hyperparameters = {}

    id_corpus = []

    def __init__(self, dataset=None, hyperparameters={}):
        """
        If a dataset is given initialize the model
        with the given data, otherwise create a blank
        model to initialize

        Parameters
        ----------
        dataset : dataset to use, optional
        hyperparameters : hyperparameters to use,
                          optional
        """
        self.builded = False
        self.trained = False
        if not dataset == None:
            self.initialize_model(dataset, hyperparameters)
        else:
            self.set_default_hyperparameters()
            self.dataset = Dataset()

    def initialize_model(self, dataset, hyperparameters):
        """
        Initialize the model with the given
        dataset and hyperparameters

        Parameters
        ----------
        dataset : dataset to use
        hyperparameters : hyperparameters to use
        """
        self.builded = False
        self.trained = False
        self.set_default_hyperparameters()
        self.set_hyperparameters(hyperparameters)
        self.dataset = dataset
        self.map_vocabulary()

    def map_vocabulary(self):
        """
        Create a dictionary to allow fast retrieving
        of a word from an Id.
        Id's are used to represent the words of
        the vocabulary
        """
        self.id2word = corpora.Dictionary(self.dataset.get_corpus())

    def set_hyperparameters(self, hyperparameters):
        """
        Set the hyperparameters

        Parameters
        ----------
        hyperparameters : dictionary
                          key = name of the hyperparameter
                          value = value of the hyperparameter
        """
        self.hyperparameters.update(hyperparameters)

    def save(self, model_path, dataset_path=None):
        """
        Save the model in a folder.
        By default the dataset is not saved

        Parameters
        ----------
        model_path : model directory path
        dataset_path : dataset path (optional)
        """
        Path(model_path).mkdir(parents=True, exist_ok=True)
        model = {}
        model["builded"] = self.builded
        model["trained"] = self.trained
        model["id_corpus"] = self.id_corpus
        model["hyperparameters"] = self.hyperparameters
        with open(model_path+"/model_data.json", 'w') as outfile:
            json.dump(model, outfile)
        self.id2word.save_as_text(model_path+"/id2word.txt")
        if not dataset_path is None:
            self.dataset.save(dataset_path)

    def load(self, model_path, dataset_path=None):
        """
        Load the model from a folder.
        By default the dataset is not loaded

        Parameters
        ----------
        model_path : model directory path
        dataset_path : dataset path (optional)
        """
        with open(model_path+"/model_data.json", 'r') as model_file:
            model = json.load(model_file)
        self.builded = model["builded"]
        self.trained = model["trained"]
        self.id_corpus = model["id_corpus"]
        self.hyperparameters = model["hyperparameters"]
        self.id2word = corpora.Dictionary.load_from_text(
            model_path+"/id2word.txt")
        if not dataset_path is None:
            self.dataset.load(dataset_path)

    def build_model(self):
        """
        Adapt the corpus to the model
        """
        self.id_corpus = [self.id2word.doc2bow(
            document) for document in self.dataset.get_corpus()]
        self.builded = True
        self.trained = False

    @abstractmethod
    def set_default_hyperparameters(self):
        """
        Set hyperparameters default values for the model
        """
        pass

    @abstractmethod
    def get_topics_terms(self, topk):
        """
        Return False if the model is not trained,
        return the topk words foreach topic otherwise
        """
        pass

    @abstractmethod
    def train_model(self):
        """
        Train the model
        """
        pass

    @abstractmethod
    def get_word_topic_weights(self):
        """
        Return None if the model is not trained,
        return the topic word matrix otherwise
        """
        pass
