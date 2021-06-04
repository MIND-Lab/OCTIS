from octis.models.model import Abstract_Model
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import octis.configuration.defaults as defaults


class NMF_scikit(Abstract_Model):

    def __init__(self, dataset, num_topics=100, init=None, alpha=0, l1_ratio=0, max_iter=100, tol=1e-4,
                 regularization='both', beta_loss='frobenius', solver='cd', use_partitions=True):
        """
        Initialize NMF model

        Parameters
        ----------
        num_topics (int) – Number of topics to extract.

        init (string, optional) – Method used to initialize the procedure.
        Default: None. Valid options:

            None: ‘nndsvd’ if n_components <= min(n_samples, n_features),
            otherwise random.

            ‘random’: non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

            ‘nndsvd’: Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

            ‘nndsvda’: NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

            ‘nndsvdar’: NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa for when
            sparsity is not desired)

        alpha (double, optional) – Constant that multiplies the regularization
        terms. Set it to zero to have no regularization.

        l1_ratio (double, optional) – The regularization mixing parameter, with
        0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an elementwise
        L2 penalty (aka Frobenius Norm). For l1_ratio = 1 it is an
        elementwise L1 penalty. For 0 < l1_ratio < 1, the penalty
        is a combination of L1 and L2.
        """
        super().__init__()
        self.hyperparameters["n_components"] = num_topics
        self.hyperparameters["init"] = init
        self.hyperparameters["alpha"] = alpha
        self.hyperparameters["l1_ratio"] = l1_ratio
        self.hyperparameters["max_iter"] = max_iter
        self.hyperparameters["tol"] = tol
        self.hyperparameters["regularization"] = regularization
        self.hyperparameters["beta_loss"] = beta_loss
        self.hyperparameters["solver"] = solver

        self.use_partitions = use_partitions
        self.update_with_test = False

        vectorizer = TfidfVectorizer(min_df=0.0, token_pattern=r"(?u)\b[\w+|\-]+\b",
                                     vocabulary=list(dataset.get_vocabulary()))

        if self.use_partitions:
            partition = dataset.get_partitioned_corpus(use_validation=False)
            corpus = partition[0]
        else:
            corpus = dataset.get_corpus()

        real_corpus = [" ".join(document) for document in corpus]
        X = vectorizer.fit_transform(real_corpus)

        self.id2word = {i: k for i, k in enumerate(vectorizer.get_feature_names())}
        if self.use_partitions:
            test_corpus = []
            for document in partition[1]:
                test_corpus.append(" ".join(document))
            Y = vectorizer.transform(test_corpus)
            self.train_corpus = X
            self.test_corpus = Y
        else:
            self.train_corpus = X

    def hyperparameters_info(self):
        """
        Returns hyperparameters informations
        """
        return defaults.NMF_scikit_hyperparameters_info

    def partitioning(self, use_partitions, update_with_test=False):
        """
        Handle the partitioning system to use and reset the model to perform
        new evaluations

        Parameters
        ----------
        use_partitions: True if train/set partitioning is needed, False
                        otherwise
        update_with_test: True if the model should be updated with the test set,
                          False otherwise
        """
        self.use_partitions = use_partitions
        self.update_with_test = update_with_test
        self.id2word = None
        self.train_corpus = None

    def train_model(self, hyperparameters=None, topics=10):
        """
        Train the model and return output

        Parameters
        ----------
        dataset : dataset to use to build the model
        hyperparameters : hyperparameters to build the model
        topics : if greather than 0 returns the most significant words
                 for each topic in the output
                 Default True


        Returns
        -------
        result : dictionary with up to 3 entries,
                 'topics', 'topic-word-matrix' and
                 'topic-document-matrix'
        """
        if hyperparameters is None:
            hyperparams = {}
        else:
            hyperparams = hyperparameters.copy()

        if "num_topics" not in hyperparams:
            hyperparams["n_components"] = self.hyperparameters["n_components"]
        else:
            hyperparams["n_components"] = hyperparams["num_topics"]
            hyperparams.pop("num_topics")

        self.hyperparameters.update(hyperparams)
        print(self.hyperparameters)
        model = NMF(**self.hyperparameters)

        W = model.fit_transform(self.train_corpus)
        #W = W / W.sum(axis=1, keepdims=True)
        H = model.components_
        #H = H / H.sum(axis=1, keepdims=True)

        result = {}

        # Find where all values in the columns are zero
        mask = (H.T == 0).all(0)
        # Update x to only include the columns where non-zero values occur.
        masked_H = H[~mask]

        result["topic-word-matrix"] = masked_H

        result["topics"] = self.get_topics(masked_H, topics)

        result["topic-document-matrix"] = np.array(W).transpose()[~mask]

        if self.use_partitions:
            if self.update_with_test:
               # NOT IMPLEMENTED YET
                result["test-topic-word-matrix"] = W
                if topics > 0:
                    result["test-topics"] = self.get_topics(W, topics)

                result["test-topic-document-matrix"] = H
            else:
                result["test-topic-document-matrix"] = model.transform(
                    self.test_corpus).T[~mask]

        return result

    def get_topics(self, H, topics):
        topic_list = []
        for topic in H:
            words_list = sorted(
                list(enumerate(topic)), key=lambda x: -x[1])
            topk = [tup[0] for tup in words_list[0:topics]]
            topic_list.append([self.id2word[i] for i in topk])
        return topic_list
