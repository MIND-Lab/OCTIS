from abc import ABC, abstractmethod

class Abstract_Model(ABC):

    word_id = {}
    id_word = {}

    def __init__(self, dataset, hyperparameters):
        self.hyperparameters = hyperparameters
        self.dataset = dataset
        self.topic_word_matrix = None
        self.topic_representation = None
        self.map_vocabulary()

    
    @abstractmethod
    def build_model(self):
       pass
    
    def map_vocabulary(self):
        vocabulary = self.dataset.get_vocabulary()
        id = 0
        for key in vocabulary.keys():
            self.word_id[key] = id
            self.id_word[id] = key
            id += 1



