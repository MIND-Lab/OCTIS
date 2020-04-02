from abc import ABC, abstractmethod

class Abstract_Model(ABC):

    def __init__(self, dataset, hyperparameters):
        self.hyperparameters = hyperparameters
        self.dataset = dataset
        self.topic_word_matrix = None
        self.topic_representation = None

    
    @abstractmethod
    def build_model(self):
       pass



