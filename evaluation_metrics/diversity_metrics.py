from evaluation_metrics.metrics import Abstract_Metric

citation = r"""
@article{DBLP:journals/corr/abs-1907-04907,
  author    = {Adji B. Dieng and
               Francisco J. R. Ruiz and
               David M. Blei},
  title     = {Topic Modeling in Embedding Spaces},
  journal   = {CoRR},
  volume    = {abs/1907.04907},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.04907},
  archivePrefix = {arXiv},
  eprint    = {1907.04907},
  timestamp = {Wed, 17 Jul 2019 10:27:36 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1907-04907.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""


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

    def info(self):
        return {
            "citation": citation,
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
