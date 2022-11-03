from octis.evaluation_metrics.metrics import AbstractMetric
from gensim.corpora.dictionary import Dictionary
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence


class TopicInterpretability(AbstractMetric):
    def __init__(
        self,  # noqa
        texts: str = None,
        topk: int = 10,
        coherence_measure: str = "c_npmi",
    ) -> None:
        """
        Initialize metric
        Parameters
        ----------
        texts : list of documents (list of lists of strings)
        topk : how many most likely words to consider in
        the evaluation
        measure : (default 'c_npmi') measure to use.
        """
        super().__init__()
        if texts is None:
            raise Exception("There are no texts in the document")
        else:
            self._texts = texts
        self._dictionary = Dictionary(self._texts)
        self.topk = topk
        self.coherence_measure = coherence_measure
        c_npmi = Coherence(texts, topk=topk, measure=coherence_measure)
        topic_diversity = TopicDiversity(topk=topk)
        self.c_npmi = c_npmi
        self.topic_diversity = topic_diversity

    def score(self, model_output: dict) -> float:  # noqa

        if self.c_npmi.score(model_output) != 0:
            # 1 is added to convert npmi output into a positive scale
            return (1+self.c_npmi.score(
                model_output
            )) * self.topic_diversity.score(model_output)

        elif self.topic_diversity.score(model_output) == 0:
            return 0
