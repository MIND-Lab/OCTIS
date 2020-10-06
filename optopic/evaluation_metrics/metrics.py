from abc import ABC, abstractmethod


class Abstract_Metric(ABC):
    """
    Class structure of a generic metric implementation
    """

    def __init__(self, metric_parameters={}):
        pass

    @abstractmethod
    def score(self):
        """
        Retrieves the score of the metric
        """
        pass
