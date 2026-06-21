import numpy as np
from scipy.special import expit


class Replicated_Softmax:
    def __init__(self):
        self.W = None

    ######  to implement in the specific class

    def train(self):
        raise NotImplementedError

    def train_epoch(self):
        raise NotImplementedError

    def set_train_hyper(self):
        raise NotImplementedError

    def visible2hidden(self):
        raise NotImplementedError

    ####### activations and sampling

    def softmax(self, x):
        """
        Softmax activation by row of the matrix x.
        The denominator log(sum(exp(x[i]))) leads to many inf, so the
        LogSumExp approximation is used instead.
        """
        maxs = np.max(x, axis=1, keepdims=True)
        lse = maxs + np.log(np.sum(np.exp(x - maxs), axis=1, keepdims=True))
        return np.exp(x - lse)

    def softmax_vec(self, array):
        """Numerically stable softmax for a single document / 1D vector (uses the same LSE trick as softmax)."""
        shifted = array - np.max(array)   # subtract max for stability
        exparr = np.exp(shifted)
        return exparr / exparr.sum()


    def sigmoid(self, x):
        """Numerically stable sigmoid activation"""
        return expit(x)


    def multinomial_sample(self, probs, N):
        """
        wrapper of np.random.multinomial
        probs: vector of probabilities for words count
        N: number of words to sample
        """
        return np.random.multinomial(N, probs, size=1)[0]

    def unif_reject_sample(self, probs):
        """
        function to sample topics (bernoulli distributed)
        given a vector of probabilities.
        It samples from a uniform distribution U(0,1)
        to get the thresholds for each topic.
        """
        h_unif = np.random.rand(*probs.shape)
        h_sample = np.array(h_unif < probs, dtype=int)
        return h_sample

    def deterministic_sample(self, probs):
        """
        function to sample topics (bernoulli distributed)
        given a vector of probabilities.
        It uses the >0.5 rule to assign 1 to each topic.
        """
        return (probs > 0.5).astype(int)

    ################### gradient utils

    def interaction_penalty(self, vel_vh, w_vh):
        """
        function to adjust the gradient of the
        topic-word interaction weights during a training iteration
        of a RS model by a penalty factor.
        The model should have the attributes:
        - penalty : bool : if the penalization should be applied
        - penL1: bool : if the penalty is of type L1 or L2
        - local_penalty : bool : if the penalty should be local or global
        - decay : float : the penalty factor to use
        This function also requires two numpy arrays as arguments:
        - the interaction weights matrix w_vh, that connects topics to words
        - the respective gradients vel_vh (also a matrix)
        """
        if self.penalty:
            if self.penL1:  # L1 penalty
                if self.local_penalty:
                    penal = self.decay * np.sign(w_vh)
                else:
                    penal = self.decay * np.sum(np.abs(w_vh)) * np.sign(w_vh)
            else:  # L2 penalty
                if self.local_penalty:
                    penal = self.decay * w_vh
                else:
                    penal = self.decay * np.sum(w_vh)

            vel_vh = vel_vh - penal
        return vel_vh

    ############### likelihood utils


    def neg_free_energy(self, v):
        """
        Given a BoW vector or document-term matrix v, computes the
        log pdf under the replicated softmax.
        Accepts both a 1D array (single document) and 2D array (batch).
        """
        w_vh, w_v, w_h = self.W
        T = self.hidden
        D = v.sum(axis=-1)  # works for both 1D and 2D
        fren = np.dot(v, w_v)
        for j in range(T):
            w_j = w_vh[:, j]
            a_j = w_h[j]
            #fren += np.log(1 + np.exp(D * a_j + np.dot(v, w_j)))
            arg = D * a_j + np.dot(v, w_j)
            fren += np.logaddexp(0, arg)   # = log(1 + exp(arg)), numerically stable
        return fren

    def marginal_pdf(self, v):
        return np.exp(self.neg_free_energy(v))

    ############ octis output functions

    def topic_words(self, topk, id2word=None):
        """
        Given a gensim dictionary id2word,
        returns the topk most important words for each topic
        inside a list of T lists, where T is the number of topics
        """
        w_vh, w_v, w_h = self.W
        T = self.hidden
        if id2word is None:
            id2word = self.id2word
        words = np.array([k for k in id2word.token2id.keys()])

        toplist = []
        for t in range(T):
            topw = w_vh[:, t]
            bestwords = words[np.argsort(topw)[::-1]][0:topk]
            toplist.append(bestwords)

        return toplist

    def _get_topics(self, topk):
        """
        Given a gensim dictionary id2word,
        returns the topk most important words for each topic
        inside a list of T lists, where T is the number of topics
        (this function is a wrapper of topic_words, used by octis class)
        """
        return self.topic_words(topk, self.id2word)

    def _get_topic_word_matrix(self):
        """
        Returns the topic representation of the words.
        Uses min-max normalization by topic of the interaction weights
        matrix w_vh. The ranking of the words using this matrix
        is equivalent to the ranking obtained from the unnormalized
        matrix of weights w_vh.
        """
        w_vh, w_v, w_h = self.W
        topic_word_matrix = w_vh.T
        normalized = []
        for words_w in topic_word_matrix:
            minimum = min(words_w)
            words = words_w - minimum
            normalized.append([float(i) / sum(words) for i in words])
        topic_word_matrix = np.array(normalized)
        return topic_word_matrix

    def _get_topic_doc(self, dtm):
        """
        given a bidimensional array dtm like, returns
        the probabilities of each topic for each document
        (as an array of probabilities).
        """
        return self.visible2hidden(dtm).T

    ####################### train utils

    def set_structure_from_dtm(
        self,
        winit=None,
        dtm=None,
        val_dtm=None,
        softstart=0.001,
        num_topics=5,
        epochs=5,
        monitor_ppl=False,
        monitor_time=False,
        monitor_loglik=False,
        logdtm=False,
    ):
        """function to initialize the weights matrices
        given the dtm and the number of topics"""
        doval = val_dtm is not None

        if logdtm:
            self.dtm = np.log(1 + dtm)
            if doval:
                self.val_dtm = np.log(1 + val_dtm)
        else:
            self.dtm = dtm
            if doval:
                self.val_dtm = val_dtm

        D = self.dtm.sum(axis=1)
        if np.any(D == 0):
            raise ValueError(
                "All training documents must have positive length; "
                f"found {(D == 0).sum()} empty document(s)."
            )
    
        if doval:
            D_val = self.val_dtm.sum(axis=1)
            if np.any(D_val == 0):
                raise ValueError(
                    "All validation documents must have positive length; "
                    f"found {(D_val == 0).sum()} empty document(s)."
                )

        self.hidden = num_topics
        self.F = num_topics
        N, dictsize = dtm.shape
        self.visible = dictsize

        self.obs_ids = np.arange(N)

        if winit is not None:
            ###self.W = winit WRONG: You are referencing the same arrays across runs
            # defensive copy to avoid sharing mutable numpy arrays across runs
            try:
                self.W = tuple(np.array(arr, copy=True) for arr in winit)
            except Exception:
                # fallback: keep original if not iterable
                self.W = winit

        if self.W is None:
            w_vh = softstart * np.random.randn(dictsize, num_topics)
            w_v = softstart * np.random.randn(dictsize)
            w_h = softstart * np.random.randn(num_topics)
            self.W = w_vh, w_v, w_h
        else:
            print("train already available weights")
            w_vh, w_v, w_h = self.W

        if monitor_time:
            self.train_time = np.empty(epochs)

        if monitor_ppl:
            self.train_ppl = np.empty(epochs)
            if doval:
                self.val_ppl = np.empty(epochs)

        if monitor_loglik:
            self.train_loglik = np.empty(epochs)
            if doval:
                self.val_loglik = np.empty(epochs)
