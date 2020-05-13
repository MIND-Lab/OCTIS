from abc import ABC, abstractmethod


class Abstract_Model(ABC):
    """
    Class structure of a generic Topic Modelling implementation
    """

    hyperparameters = {}

    def __init__(self):
        """
        Create a blank model to initialize
        """

    @abstractmethod
    def train_model(self, dataset, hyperparameters):
        """
        Train the model.
        Return a dictionary with up to 3 entries,
        'topics', 'topic-word-matrix' and 'topic-document-matrix'.
        'topics' is the list of the most significative words for
        each topic (list of lists of strings).
        'topic-word-matrix' is a matrix NxM where N is the number
        of topics and M is the vocabulary length.
        'topi-document-matrix' is a matrix NxM where N is the number
        of topics and M is the number of documents in the corpus.

        """
        pass
