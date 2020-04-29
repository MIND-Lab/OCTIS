from evaluation_metrics.metrics import Abstract_Metric


class Topic_diversity(Abstract_Metric):
    def __init__(self, metric_parameters={'topk': 10}):
        """
        Initialize metric

        Parameters
        ----------
        metric_parameters : dictionary with key 'topk'
                            topk: top k words on which the topic diversity
                            will be computed
        """
        self.topk = metric_parameters["topk"]

    def score(self, model_output):
        """
        Retrieves the score of the metric

        Parameters
        ----------
        model_output : output of the model in the format
                       [topics, topic word matrix, topic document matrix]
                       topics required.

        Returns
        -------
        td : score
        """
        self.topics = model_output[0]
        if self.topk > len(self.topics[0]):
            raise Exception('Words in topics are less than '+str(self.topk))
        else:
            unique_words = set()
            for topic in self.topics:
                unique_words = unique_words.union(set(topic[:self.topk]))
            td = len(unique_words) / (self.topk * len(self.topics))
            return td
