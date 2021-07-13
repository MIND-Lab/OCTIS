from octis.models.model import AbstractModel
from gensim.models import lsimodel
import numpy as np
import gensim.corpora as corpora
import octis.configuration.defaults as defaults


class LSI(AbstractModel):

    id2word = None
    id_corpus = None
    hyperparameters = {}
    use_partitions = True
    update_with_test = False

    def __init__(self, num_topics=200, chunksize=20000, decay=1.0,
                 distributed=False, onepass=True, power_iters=2,
                 extra_samples=100):
        """
        Initialize LSI model

        Parameters
        ----------
        num_topics (int, optional) – Number of requested factors

        chunksize (int, optional) – Number of documents to be used in each
        training chunk.

        decay (float, optional) – Weight of existing observations relatively
        to new ones.

        distributed (bool, optional) – If True - distributed mode (parallel
        execution on several machines) will be used.

        onepass (bool, optional) – Whether the one-pass algorithm should be
        used for training. Pass False to force a multi-pass stochastic algorithm.

        power_iters (int, optional) – Number of power iteration steps to be used.
        Increasing the number of power iterations improves accuracy, but lowers
        performance

        extra_samples (int, optional) – Extra samples to be used besides the rank
        k. Can improve accuracy.
        """
        self.hyperparameters["num_topics"] = num_topics
        self.hyperparameters["chunksize"] = chunksize
        self.hyperparameters["decay"] = decay
        self.hyperparameters["distributed"] = distributed
        self.hyperparameters["onepass"] = onepass
        self.hyperparameters["power_iters"] = power_iters
        self.hyperparameters["extra_samples"] = extra_samples

    def info(self):
        """
        Returns model informations
        """
        return {
            "name": "LSI, Latent Semantic Indexing"
        }

    def hyperparameters_info(self):
        """
        Returns hyperparameters informations
        """
        return defaults.LSI_hyperparameters_info

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

    def train_model(self, dataset, hyperparameters={}, top_words=10):
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
        partition = []
        if self.use_partitions:
            partition = dataset.get_partitioned_corpus(use_validation=False)
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

        self.trained_model = lsimodel.LsiModel(**self.hyperparameters)

        result = {}

        result["topic-word-matrix"] = self._get_topic_word_matrix()

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
                self.trained_model.add_documents(new_corpus)
                self.id_corpus.extend(new_corpus)

                result["test-topic-word-matrix"] = self._get_topic_word_matrix()

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
                test_document_topic_matrix = []
                for document in new_corpus:

                    document_topics_tuples = self.trained_model[document]
                    document_topics = np.zeros(
                        self.hyperparameters["num_topics"])
                    for single_tuple in document_topics_tuples:
                        document_topics[single_tuple[0]] = single_tuple[1]

                    test_document_topic_matrix.append(document_topics)

                result["test-topic-document-matrix"] = np.array(
                    test_document_topic_matrix).transpose()

        return result

    def _get_topic_word_matrix(self):
        """
        Return the topic representation of the words
        """
        topic_word_matrix = self.trained_model.get_topics()
        normalized = []
        for words_w in topic_word_matrix:
            minimum = min(words_w)
            words = words_w - minimum
            normalized.append([float(i)/sum(words) for i in words])
        topic_word_matrix = np.array(normalized)
        return topic_word_matrix

    def _get_topics_words(self, topics):
        """
        Return the most significative words for each topic.
        """
        topic_terms = []
        for i in range(self.hyperparameters["num_topics"]):
            topic_words_list = []
            for word_tuple in self.trained_model.show_topic(i, topics):
                topic_words_list.append(word_tuple[0])
            topic_terms.append(topic_words_list)
        return topic_terms

    def _get_topic_document_matrix(self):
        """
        Return the topic representation of the
        corpus
        """
        topic_weights = self.trained_model[self.id_corpus]

        topic_document = []

        for document_topic_weights in topic_weights:

            # Find min e max topic_weights values
            minimum = document_topic_weights[0][1]
            maximum = document_topic_weights[0][1]
            for topic in document_topic_weights:
                if topic[1] > maximum:
                    maximum = topic[1]
                if topic[1] < minimum:
                    minimum = topic[1]

            # For each topic compute normalized weight
            # in the form (value-min)/(max-min)
            topic_w = []
            for topic in document_topic_weights:
                topic_w.append((topic[1]-minimum)/(maximum-minimum))
            topic_document.append(topic_w)

        return np.array(topic_document).transpose()
