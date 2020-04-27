from evaluation_metrics.metrics import Abstract_Metric


class Topic_diversity(Abstract_Metric):
    def __init__(self, model_output,  metric_parameters={'topk':10}):
        """
        Initialize metric

        Parameters
        ----------
        model_output : output of the model in the format
                       [topics, topic word matrix, topic document matrix]
                       topics required.
        metric_parameters : dictionary with key 'topk'
                            topk: top k words on which the topic diversity
                            will be computed
        """
        self.topics = model_output[0]
        self.topk = metric_parameters["topk"]

    def score(self):
        """
        Retrieves the score of the metric

        Returns
        -------
        td : score
        """
        if self.topk > len(self.topics[0]):
            raise Exception('Words in topics are less than '+str(self.topk))
        else:
            unique_words = set()
            for topic in self.topics:
                unique_words = unique_words.union(set(topic[:self.topk]))
            td = len(unique_words) / (self.topk * len(self.topics))
            return td
