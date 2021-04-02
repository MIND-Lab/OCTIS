from abc import ABC, abstractmethod


class AbstractMetric(ABC):
    """
    Class structure of a generic metric implementation
    """

    def __init__(self, metric_parameters=None):
        """
        init metric

        :param metric_parameters: parameters of a generic metric
        :type dict {parameter name: value}
        """
        if metric_parameters is None:
            self.metric_parameters = {}

    @abstractmethod
    def score(self, model_output):
        """
        Retrieves the score of the metric

        :param model_output: output of a topic model in the form of a dictionary. See model for details on
        the model output
        :type model_output: dict
        """
        pass
