from models.model import Abstract_Model
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import DictionaryLearning
from gensim.models import nmf
import gensim.corpora as corpora
import configuration.citations as citations
import configuration.defaults as defaults
import scipy.sparse


class NMF_gensim(Abstract_Model):

    id2word = None
    id_corpus = None
    use_partitions = True
    update_with_test = False

    def __init__(self, num_topics=100, chunksize=2000, passes=1, kappa=1.0,
                 minimum_probability=0.01, w_max_iter=200,
                 w_stop_condition=0.0001, h_max_iter=50, h_stop_condition=0.001,
                 eval_every=10, normalize=True, random_state=None):
        """
        Initialize NMF model

        Parameters
        ----------
        num_topics (int, optional) – Number of topics to extract.

        chunksize (int, optional) – Number of documents to be used in each 
        training chunk.

        passes (int, optional) – Number of full passes over the
        training corpus. Leave at default passes=1 if your input
        is an iterator.

        kappa (float, optional) – Gradient descent step size.
        Larger value makes the model train faster, but could
        lead to non-convergence if set too large.

        minimum_probability – If normalize is True, topics with
        smaller probabilities are filtered out. If normalize is False,
        topics with smaller factors are filtered out. If set to None,
        a value of 1e-8 is used to prevent 0s.

        w_max_iter (int, optional) – Maximum number of iterations to
        train W per each batch.

        w_stop_condition (float, optional) – If error difference gets less
        than that, training of W stops for the current batch.

        h_max_iter (int, optional) – Maximum number of iterations to train 
        h per each batch.

        h_stop_condition (float) – If error difference gets less than that,
        training of h stops for the current batch.

        eval_every (int, optional) – Number of batches after which l2 norm
        of (v - Wh) is computed. Decreases performance if set too low.

        normalize (bool or None, optional) – Whether to normalize the result.

        random_state ({np.random.RandomState, int}, optional) – Seed for
        random generator. Needed for reproducibility.
        """
        self.hyperparameters["num_topics"] = num_topics
        self.hyperparameters["chunksize"] = chunksize
        self.hyperparameters["passes"] = passes
        self.hyperparameters["kappa"] = kappa
        self.hyperparameters["minimum_probability"] = minimum_probability
        self.hyperparameters["w_max_iter"] = w_max_iter
        self.hyperparameters["w_stop_conditiom"] = w_stop_condition
        self.hyperparameters["h_max_iter"] = h_max_iter
        self.hyperparameters["h_stop_condition"] = h_stop_condition
        self.hyperparameters["eval_every"] = eval_every
        self.hyperparameters["normalize"] = normalize
        self.hyperparameters["random_state"] = random_state

    def info(self):
        """
        Returns model informations
        """
        return {
            "citation": citations.models_NMF,
            "name": "NMF, Non-negative Matrix Factorization"
        }

    def hyperparameters_info(self):
        """
        Returns hyperparameters informations
        """
        return defaults.NMF_gensim_hyperparameters_info

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
        self.id_corpus = None

    def train_model(self, dataset, hyperparameters={}, top_words=10,
                    topic_word_matrix=True, topic_document_matrix=True):
        """
        Train the model and return output

        Parameters
        ----------
        dataset : dataset to use to build the model
        hyperparameters : hyperparameters to build the model
        top_words : if greather than 0 returns the most significant words
                 for each topic in the output
                 Default True
        topic_word_matrix : if True returns the topic word matrix in the output
                            Default True
        topic_document_matrix : if True returns the topic document
                                matrix in the output
                                Default True

        Returns
        -------
        result : dictionary with up to 3 entries,
                 'topics', 'topic-word-matrix' and
                 'topic-document-matrix'
        """
        partition = []
        if self.use_partitions:
            partition = dataset.get_partitioned_corpus()
        else:
            partition = [dataset.get_corpus(), []]

        if self.id2word == None:
            self.id2word = corpora.Dictionary(dataset.get_corpus())

        if self.id_corpus == None:
            self.id_corpus = [self.id2word.doc2bow(
                document) for document in partition[0]]

        hyperparameters["corpus"] = self.id_corpus
        hyperparameters["id2word"] = self.id2word
        self.hyperparameters.update(hyperparameters)

        self.trained_model = nmf.Nmf(**self.hyperparameters)

        result = {}

        if topic_word_matrix:
            result["topic-word-matrix"] = self.trained_model.get_topics()

        if top_words > 0:
            topics_output = []
            for topic in result["topic-word-matrix"]:
                top_k = np.argsort(topic)[-top_words:]
                top_k_words = list(reversed([self.id2word[i] for i in top_k]))
                topics_output.append(top_k_words)
            result["topics"] = topics_output

        if topic_document_matrix:
            result["topic-document-matrix"] = self._get_topic_document_matrix()

        if self.use_partitions:
            new_corpus = [self.id2word.doc2bow(
                document) for document in partition[1]]
            if self.update_with_test:
                self.trained_model.update(new_corpus)
                self.id_corpus.extend(new_corpus)

                if topic_word_matrix:
                    result["test-topic-word-matrix"] = self.trained_model.get_topics()

                if top_words > 0:
                    topics_output = []
                    for topic in result["test-topic-word-matrix"]:
                        top_k = np.argsort(topic)[-top_words:]
                        top_k_words = list(
                            reversed([self.id2word[i] for i in top_k]))
                        topics_output.append(top_k_words)
                    result["test-topics"] = topics_output

                if topic_document_matrix:
                    result["test-topic-document-matrix"] = self._get_topic_document_matrix()

            else:
                test_document_topic_matrix = []
                for document in new_corpus:
                    test_document_topic_matrix.append(
                        self.trained_model[document])
                result["test-document-topic-matrix"] = np.array(
                    test_document_topic_matrix)

        return result

    def _get_topics_words(self, topk):
        """
        Return the most significative words for each topic.
        """
        topic_terms = []
        for i in range(self.hyperparameters["num_topics"]):
            topic_words_list = []
            for word_tuple in self.trained_model.get_topic_terms(i, topk):
                topic_words_list.append(self.id2word[word_tuple[0]])
            topic_terms.append(topic_words_list)
        return topic_terms

    def _get_topic_document_matrix(self):
        """
        Return the topic representation of the
        corpus
        """
        doc_topic_tuples = []
        for document in self.id_corpus:
            doc_topic_tuples.append(
                self.trained_model.get_document_topics(document,
                                                       minimum_probability=0))

        topic_document = np.zeros((
            self.hyperparameters["num_topics"],
            len(doc_topic_tuples)))

        for ndoc in range(len(doc_topic_tuples)):
            document = doc_topic_tuples[ndoc]
            for topic_tuple in document:
                topic_document[topic_tuple[0]][ndoc] = topic_tuple[1]
        return topic_document


class NMF_scikit(Abstract_Model):

    id2word = None
    id_corpus = None
    use_partitions = True
    update_with_test = False
    hyperparameters = {"num_topics": 100,
                       "init": "random", "alpha": 0, "l1_ratio": 0}

    def __init__(self, num_topics=100, init=None, alpha=0, l1_ratio=0):
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
        self.hyperparameters["num_topics"] = num_topics
        self.hyperparameters["init"] = init
        self.hyperparameters["alpha"] = alpha
        self.hyperparameters["l1_ratio"] = l1_ratio

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
        self.id_corpus = None

    def train_model(self, dataset, hyperparameters={}, topics=10,
                    topic_word_matrix=True, topic_document_matrix=True):
        """
        Train the model and return output

        Parameters
        ----------
        dataset : dataset to use to build the model
        hyperparameters : hyperparameters to build the model
        topics : if greather than 0 returns the most significant words
                 for each topic in the output
                 Default True
        topic_word_matrix : if True returns the topic word matrix in the output
                            Default True
        topic_document_matrix : if True returns the topic document
                                matrix in the output
                                Default True

        Returns
        -------
        result : dictionary with up to 3 entries,
                 'topics', 'topic-word-matrix' and
                 'topic-document-matrix'
        """
        if self.id2word == None or self.id_corpus == None:
            vectorizer = TfidfVectorizer(min_df=0.0)
            corpus = dataset.get_corpus()
            real_corpus = []
            for document in corpus:
                real_corpus.append(" ".join(document))
            X = vectorizer.fit_transform(real_corpus)

            lista = vectorizer.get_feature_names()
            self.id2word = {i: lista[i] for i in range(0, len(lista))}
            if self.use_partitions:
                ltd = dataset.get_metadata()[
                    "last-training-doc"]
                self.id_corpus = X[0:ltd]
                self.new_corpus = X[ltd:]
            else:
                self.id_corpus = X

        hyperparameters["corpus"] = self.id_corpus
        hyperparameters["id2word"] = self.id2word
        self.hyperparameters.update(hyperparameters)

        model = NMF(
            n_components=self.hyperparameters["num_topics"],
            init=self.hyperparameters["init"],
            alpha=self.hyperparameters["alpha"],
            l1_ratio=self.hyperparameters["l1_ratio"])

        W = model.fit_transform(self.id_corpus)
        #W = W / W.sum(axis=1, keepdims=True)
        H = model.components_
        #H = H / H.sum(axis=1, keepdims=True)

        result = {}

        if topic_word_matrix:
            result["topic-word-matrix"] = H

        if topics > 0:
            result["topics"] = self.get_topics(H, topics)

        if topic_document_matrix:
            result["topic-document-matrix"] = np.array(W).transpose()

        if self.use_partitions:
            if self.update_with_test:
               # NOT IMPLEMENTED YET

                if topic_word_matrix:
                    result["test-topic-word-matrix"] = W

                if topics > 0:
                    result["test-topics"] = self.get_topics(W, topics)

                if topic_document_matrix:
                    result["test-topic-document-matrix"] = H

            else:
                result["test-document-topic-matrix"] = model.transform(
                    self.new_corpus)

        return result

    def get_topics(self, H, topics):
        topic_list = []
        for topic in H:
            words_list = sorted(
                list(enumerate(topic)), key=lambda x: x[1])
            topk = [tup[0] for tup in words_list[0:topics]]
            topic_list.append([self.id2word[i] for i in topk])
        return topic_list


def NMF_Model(implementation="gensim"):
    """
    Choose NMF implementation and return the correct model

    Parameters
    ----------
    implementation: implementation of NMF to use
                    available 'gensim' and 'scikit'
                    default 'gensim'

    Returns
    -------
    model: an initialized model of the choosen implementation
    """
    if implementation == "gensim":
        return NMF_gensim()
    if implementation == "scikit":
        return NMF_scikit()