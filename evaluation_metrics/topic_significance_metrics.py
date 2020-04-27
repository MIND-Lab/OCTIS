import numpy as np

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


class KL_uniform(Abstract_Metric):
    def __init__(self, model_output):
        """
        Initialize metric

        Parameters
        ----------
        model_output : output of the model in the format
                       [topics, topic word matrix, topic document matrix]
                       distribution of topics over words matrix
                       phi[topic][word] required
        """
        self.phi = model_output[1]

    def score(self):
        """
        Retrieves the score of the metric

        Returns
        -------
        result : score
        """
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
    def __init__(self, model_output):
        """
        Initialize metric

        Parameters
        ----------
        model_output : output of the model in the format
                       [topics, topic word matrix, topic document matrix]
                       distribution of topics over words matrix
                       phi[topic][word] required.
                       distribution of topics over documents matrix
                       theta[topic][document] required.
        """
        self.phi = model_output[1]
        self.theta = model_output[2]

    def score(self):
        """
        Retrieves the score of the metric

        Returns
        -------
        result : score
        """
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
    def __init__(self, model_output):
        """
        Initialize metric

        Parameters
        ----------
        model_output : output of the model in the format
                       [topics, topic word matrix, topic document matrix]
                       distribution of topics over documents matrix
                       theta[topic][document] required.
        """
        self.theta = model_output[2]

    def score(self):
        """
        Retrieves the score of the metric

        Returns
        -------
        result : score
        """
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
