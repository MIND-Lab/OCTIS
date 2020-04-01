from abc import ABC, abstractmethod

class Abstract_Model(ABC):

    def __init__(self, dataset, hyperparameters):
        self.hyperparameters = hyperparameters
        self.dataset = dataset
        self.topic_word_matrix = None

    
    @abstractmethod
    def model_builder(self, dataset, hyperparameters):
        pass

    def build_model(self):
        self.topic_word_matrix = self.model_builder(
            self.dataset,
            self.hyperparameters)



