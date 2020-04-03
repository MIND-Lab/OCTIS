from abc import ABC, abstractmethod


class Abstract_Model(ABC):
    """
    Interface of a generic Topic Modelling implementation
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

    topic_word_matrix = []
    doc_topic_representation = []

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

    def initialize_model(self, dataset, hyperparameters):
        """
        Initialize the model with the given
        dataset and hyperparameters

        Parameters
        ----------
        dataset : dataset to use,
                  default = no dataset
        hyperparameters : hyperparameters to use
                          optional
        """
        self.builded = False
        self.trained = False
        self.set_default_hyperparameters()
        self.set_hyperparameters(hyperparameters)
        self.dataset = dataset
        self.topic_word_matrix = None
        self.topic_representation = None
        self.map_vocabulary()

    @abstractmethod
    def set_default_hyperparameters(self):
        """
        Set hyperparameters default values for the model
        """
        pass

    @abstractmethod
    def build_model(self):
        """
        Adapt the corpus to the model
        """
        pass

    @abstractmethod
    def train_model(self):
        """
        Train the model
        """
        pass

    @abstractmethod
    def make_doc_topic_representation(self):
        """
        Return False if the model is not trained,
        produce the document topic representation
        and return True otherwise
        """
        pass

    @abstractmethod
    def make_topic_word_matrix(self):
        """
        Return False if the model is not trained,
        produce the topic word matrix and return
        True otherwise
        """
        pass

    @abstractmethod
    def map_vocabulary(self):
        """
        Create a dictionary to allow fast retrieving
        of a word from an Id.
        Id's are used to represent the words of
        the vocabulary
        """
        pass


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
