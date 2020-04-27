from evaluation_metrics.metrics import Abstract_Metric


class Topic_diversity(Abstract_Metric):
    def __init__(self, topics):
        """
        Initialize metric

        Parameters
        ----------
        topics : lists of the words of each topic
        """
        self.topics = topics

    def score(self, topk=25):
        """
        Retrieves the score of the metric

        Parameters
        ----------
        topk : top k words on which the topic diversity
               will be computed

        Returns
        -------
        td : score
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than '+str(topk))
        else:
            unique_words = set()
            for topic in self.topics:
                unique_words = unique_words.union(set(topic[:topk]))
            td = len(unique_words) / (topk * len(self.topics))
            return td
