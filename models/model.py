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
        Train the model
        """
        pass
