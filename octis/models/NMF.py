from octis.models.model import AbstractModel
import numpy as np
from gensim.models import nmf
import gensim.corpora as corpora
import octis.configuration.citations as citations
import octis.configuration.defaults as defaults


class NMF(AbstractModel):

    def __init__(self, num_topics=100, chunksize=2000, passes=1, kappa=1.0, minimum_probability=0.01, w_max_iter=200,
                 w_stop_condition=0.0001, h_max_iter=50, h_stop_condition=0.001, eval_every=10, normalize=True,
                 random_state=None, use_partitions=True):
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
        super().__init__()
        self.hyperparameters["num_topics"] = num_topics
        self.hyperparameters["chunksize"] = chunksize
        self.hyperparameters["passes"] = passes
        self.hyperparameters["kappa"] = kappa
        self.hyperparameters["minimum_probability"] = minimum_probability
        self.hyperparameters["w_max_iter"] = w_max_iter
        self.hyperparameters["w_stop_condition"] = w_stop_condition
        self.hyperparameters["h_max_iter"] = h_max_iter
        self.hyperparameters["h_stop_condition"] = h_stop_condition
        self.hyperparameters["eval_every"] = eval_every
        self.hyperparameters["normalize"] = normalize
        self.hyperparameters["random_state"] = random_state
        self.use_partitions = use_partitions

        self.id2word = None
        self.id_corpus = None
        self.update_with_test = False

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

    def train_model(self, dataset, hyperparameters=None, top_words=10):
        """
        Train the model and return output

        Parameters
        ----------
        dataset : dataset to use to build the model
        hyperparameters : hyperparameters to build the model
        top_words : if greather than 0 returns the most significant words
                 for each topic in the output
                 Default True

        Returns
        -------
        result : dictionary with up to 3 entries,
                 'topics', 'topic-word-matrix' and
                 'topic-document-matrix'
        """
        if hyperparameters is None:
            hyperparameters = {}
        if self.use_partitions:
            partition = dataset.get_partitioned_corpus(use_validation=False)
        else:
            partition = [dataset.get_corpus(), []]

        if self.id2word is None:
            self.id2word = corpora.Dictionary(dataset.get_corpus())
        if self.id_corpus is None:
            self.id_corpus = [self.id2word.doc2bow(
                document) for document in partition[0]]

        hyperparameters["corpus"] = self.id_corpus
        hyperparameters["id2word"] = self.id2word
        self.hyperparameters.update(hyperparameters)

        self.trained_model = nmf.Nmf(**self.hyperparameters)

        result = {}

        result["topic-word-matrix"] = self.trained_model.get_topics()

        if top_words > 0:
            topics_output = []
            for topic in result["topic-word-matrix"]:
                top_k = np.argsort(topic)[-top_words:]
                top_k_words = list(reversed([self.id2word[i] for i in top_k]))
                topics_output.append(top_k_words)
            result["topics"] = topics_output

        result["topic-document-matrix"] = self._get_topic_document_matrix()

        if self.use_partitions:
            new_corpus = [self.id2word.doc2bow(
                document) for document in partition[1]]
            if self.update_with_test:
                self.trained_model.update(new_corpus)
                self.id_corpus.extend(new_corpus)

                result["test-topic-word-matrix"] = self.trained_model.get_topics()

                if top_words > 0:
                    topics_output = []
                    for topic in result["test-topic-word-matrix"]:
                        top_k = np.argsort(topic)[-top_words:]
                        top_k_words = list(
                            reversed([self.id2word[i] for i in top_k]))
                        topics_output.append(top_k_words)
                    result["test-topics"] = topics_output

                result["test-topic-document-matrix"] = self._get_topic_document_matrix()
            else:
                result["test-topic-document-matrix"] = self._get_topic_document_matrix(new_corpus)
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

    def _get_topic_document_matrix(self, test_corpus=None):
        """
        Return the topic representation of the
        corpus
        """
        doc_topic_tuples = []

        if test_corpus is None:
            for document in self.id_corpus:
                doc_topic_tuples.append(
                    self.trained_model.get_document_topics(document, minimum_probability=0))
        else:
            for document in test_corpus:
                doc_topic_tuples.append(
                    self.trained_model.get_document_topics(document, minimum_probability=0))
        topic_document = np.zeros((
            self.hyperparameters["num_topics"],
            len(doc_topic_tuples)))

        for ndoc in range(len(doc_topic_tuples)):
            document = doc_topic_tuples[ndoc]
            for topic_tuple in document:
                topic_document[topic_tuple[0]][ndoc] = topic_tuple[1]
        return topic_document
