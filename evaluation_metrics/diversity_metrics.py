from evaluation_metrics.metrics import Abstract_Metric
import configuration.citations as citations
import configuration.defaults as defaults


class Topic_diversity(Abstract_Metric):
    def __init__(self, metric_parameters=defaults.em_topic_diversity.copy()):
        """
        Initialize metric

        Parameters
        ----------
        metric_parameters : dictionary with key 'topk'
                            topk: top k words on which the topic diversity
                            will be computed
        """
        self.topk = metric_parameters["topk"]

    def info(self):
        return {
            "citation": citations.em_topic_diversity,
            "name": "Topic diversity"
        }

    def score(self, model_output):
        """
        Retrieves the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model 
                       key 'topics' required.

        Returns
        -------
        td : score
        """
        self.topics = model_output["topics"]
        if self.topk > len(self.topics[0]):
            raise Exception('Words in topics are less than '+str(self.topk))
        else:
            unique_words = set()
            for topic in self.topics:
                unique_words = unique_words.union(set(topic[:self.topk]))
            td = len(unique_words) / (self.topk * len(self.topics))
            return td
