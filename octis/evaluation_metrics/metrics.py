from abc import ABC, abstractmethod


class Abstract_Metric(ABC):
    """
    Class structure of a generic metric implementation
    """

    def __init__(self, metric_parameters=None):
        if metric_parameters is None:
            metric_parameters = {}

    @abstractmethod
    def score(self, model_output):
        """
        Retrieves the score of the metric
        """
        pass
