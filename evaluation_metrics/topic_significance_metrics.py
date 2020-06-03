import numpy as np

from evaluation_metrics.metrics import Abstract_Metric


citation = """
@inproceedings{DBLP:conf/pkdd/AlSumaitBGD09,
  author    = {Loulwah AlSumait and
               Daniel Barbar{\'{a}} and
               James Gentle and
               Carlotta Domeniconi},
  editor    = {Wray L. Buntine and
               Marko Grobelnik and
               Dunja Mladenic and
               John Shawe{-}Taylor},
  title     = {Topic Significance Ranking of {LDA} Generative Models},
  booktitle = {Machine Learning and Knowledge Discovery in Databases, European Conference,
               {ECML} {PKDD} 2009, Bled, Slovenia, September 7-11, 2009, Proceedings,
               Part {I}},
  series    = {Lecture Notes in Computer Science},
  volume    = {5781},
  pages     = {67--82},
  publisher = {Springer},
  year      = {2009},
  url       = {https://doi.org/10.1007/978-3-642-04180-8\_22},
  doi       = {10.1007/978-3-642-04180-8\_22},
  timestamp = {Tue, 14 May 2019 10:00:47 +0200},
  biburl    = {https://dblp.org/rec/conf/pkdd/AlSumaitBGD09.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""


def _KL(P, Q):
    """
    Perform Kullback-Leibler divergence

    Parameters
    ----------
    P : distribution P
    Q : distribution Q

    Returns
    -------
    divergence : divergence from Q to P
    """
    # add epsilon to grant absolute continuity
    epsilon = 0.00001
    P = P+epsilon
    Q = Q+epsilon

    divergence = np.sum(P*np.log(P/Q))
    return divergence


class KL_uniform(Abstract_Metric):
    def __init__(self, metric_parameters={}):
        """
        Initialize metric
        """
        super().__init__()

    def info(self):
        return {
            "citation": citation,
            "name": "KL_Uniform, Uniform distribution over words"
        }

    def score(self, model_output):
        """
        Retrieves the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       'topic-word-matrix' required

        Returns
        -------
        result : score
        """
        self.phi = model_output["topic-word-matrix"]

        # make uniform distribution
        val = 1.0 / len(self.phi[0])
        unif_distr = np.full(len(self.phi[0]), val)

        divergences = []
        for topic in range(len(self.phi)):

            # normalize phi, sum up to 1
            P = self.phi[topic] / self.phi[topic].sum()

            divergence = _KL(P, unif_distr)
            divergences.append(divergence)

        # KL-uniform = mean of the divergences
        # between topic-word distributions and uniform distribution
        result = np.array(divergences).mean()
        return result


class KL_vacuous(Abstract_Metric):
    def __init__(self, metric_parameters={}):
        """
        Initialize metric
        """
        super().__init__()

    def info(self):
        return {
            "citation": citation,
            "name": "KL_Vacuous, Vacuous semantic distribution"
        }

    def score(self, model_output):
        """
        Retrieves the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       'topic-word-matrix' required
                       'topic-document-matrix' required

        Returns
        -------
        result : score
        """
        self.phi = model_output["topic-word-matrix"]
        self.theta = model_output["topic-document-matrix"]

        vacuous = np.zeros(self.phi.shape[1])
        for topic in range(len(self.theta)):

            # get probability of the topic in the corpus
            p_topic = self.theta[topic].sum()/len(self.theta[0])

            # get probability of the words:
            # P(Wi | vacuous_dist) = P(Wi | topic)*P(topic)
            vacuous += self.phi[topic]*p_topic

        divergences = []
        for topic in range(len(self.phi)):

            # normalize phi, sum up to 1
            P = self.phi[topic] / self.phi[topic].sum()

            divergence = _KL(P, vacuous)
            divergences.append(divergence)

        # KL-vacuous = mean of the divergences
        # between topic-word distributions and vacuous distribution
        result = np.array(divergences).mean()
        return result


class KL_background(Abstract_Metric):
    def __init__(self, metric_parameters={}):
        """
        Initialize metric
        """
        super().__init__()

    def info(self):
        return {
            "citation": citation,
            "name": "KL_Background, Background distribution over documents"
        }

    def score(self, model_output):
        """
        Retrieves the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       'topic-document-matrix' required

        Returns
        -------
        result : score
        """
        self.theta = model_output["topic-document-matrix"]

        # make uniform distribution
        val = 1.0 / len(self.theta[0])
        unif_distr = np.full(len(self.theta[0]), val)

        divergences = []
        for topic in range(len(self.theta)):

            # normalize theta, sum up to 1
            P = self.theta[topic] / self.theta[topic].sum()

            divergence = _KL(P, unif_distr)
            divergences.append(divergence)

        # KL-background = mean of the divergences
        # between topic-doc distributions and uniform distribution
        result = np.array(divergences).mean()
        return result
