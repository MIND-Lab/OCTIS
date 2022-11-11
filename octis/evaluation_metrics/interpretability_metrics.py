from octis.evaluation_metrics.metrics import AbstractMetric
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence,_load_default_texts


class TopicInterpretability(AbstractMetric):
    def __init__(
        self,  # noqa
        texts: str = None,
        topk: int = 10
    ) -> None:
        """
        Initialize metric
        Parameters
        ----------
        texts : list of documents (list of lists of strings)
        topk : how many most likely words to consider in
        the evaluation
        """
        super().__init__()
        if texts is None:
            self._texts = _load_default_texts()
        else:
            self._texts = texts
        self.topk=topk
        self.c_npmi = Coherence(texts, topk=topk, measure='c_npmi')
        self.topic_diversity = TopicDiversity(topk=topk)

    def score(self, model_output: dict) -> float:  # noqa
            # 1 is added to convert npmi output into a positive scale
            return (1+self.c_npmi.score(
                model_output
            )) * self.topic_diversity.score(model_output)
