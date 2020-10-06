from models.model import Abstract_Model

class CRTM_Model(Abstract_Model):
    """
    Class structure of a generic Topic Modelling implementation
    """

    hyperparameters = {}

    def __init__(self):
        """
        Create a blank model to initialize
        """

    def train_model(self, dataset, hyperparameters):
        """
        Train the model.
        Return a dictionary with up to 3 entries,
        'topics', 'topic-word-matrix' and 'topic-document-matrix'.
        'topics' is the list of the most significative words for
        each topic (list of lists of strings).
        'topic-word-matrix' is an NxV matrix of weights where N is the number
        of topics and V is the vocabulary length.
        'topic-document-matrix' is an NxD matrix of weights where N is the number
        of topics and D is the number of documents in the corpus.

        """
        pass
