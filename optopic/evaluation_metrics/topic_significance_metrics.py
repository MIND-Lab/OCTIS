import numpy as np
import configuration.citations as citations
from evaluation_metrics.metrics import Abstract_Metric


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


def _replace_zeros_lines(arr):
    zero_lines = np.where(~arr.any(axis=1))[0]
    val = 1.0 / len(arr[0])
    vett = np.full(len(arr[0]), val)
    for zero_line in zero_lines:
        arr[zero_line] = vett.copy()
    return arr


class KL_uniform(Abstract_Metric):
    def __init__(self, metric_parameters={}):
        """
        Initialize metric
        """
        super().__init__()

    def info(self):
        return {
            "citation": citations.em_topic_significance,
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
        self.phi = _replace_zeros_lines(
            model_output["topic-word-matrix"].astype(float))

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
            "citation": citations.em_topic_significance,
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
        self.phi = _replace_zeros_lines(
            model_output["topic-word-matrix"].astype(float))
        self.theta = _replace_zeros_lines(
            model_output["topic-document-matrix"].astype(float))

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
            "citation": citations.em_topic_significance,
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
        self.theta = _replace_zeros_lines(
            model_output["topic-document-matrix"].astype(float))

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
        if np.isnan(result):
            return 0
        return result