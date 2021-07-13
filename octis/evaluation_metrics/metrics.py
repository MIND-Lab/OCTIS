from abc import ABC, abstractmethod


class AbstractMetric(ABC):
    """
    Class structure of a generic metric implementation
    """

    def __init__(self):
        """
        init metric
        """
        pass

    @abstractmethod
    def score(self, model_output):
        """
        Retrieves the score of the metric

        :param model_output: output of a topic model in the form of a dictionary. See model for details on
        the model output
        :type model_output: dict
        """
        pass

    def get_params(self):
        return [att for att in dir(self) if not att.startswith("_") and att != 'info' and att != 'score' and
                att != 'get_params']
