import numpy as np
import tomotopy as tp

from octis.models.model import AbstractModel

"""
Experimental integration of tomotopy's implementation of LDA
"""


class LDA_tomopy(AbstractModel):

    def __init__(self, dataset, num_topics=100, alpha=0.1, eta=0.01, max_iters=50, use_partitions=True):
        super().__init__()
        self.hyperparameters = dict()
        self.hyperparameters['num_topics'] = num_topics
        self.hyperparameters['alpha'] = alpha
        self.hyperparameters['eta'] = eta
        self.hyperparameters['max_iters'] = max_iters
        self.use_partitions = use_partitions

        if self.use_partitions:
            self.x_train, self.x_test = dataset.get_partitioned_corpus(use_validation=False)
        else:
            self.x_train = dataset.get_corpus()
            self.x_test = None
        self.vocabulary = dataset.get_vocabulary()

    def train_model(self, hyperparameters=None, top_words=10):
        if hyperparameters is None:
            hyperparameters = dict()

        self.set_default_hyperparameters(hyperparameters)
        lda = tp.LDAModel(k=self.hyperparameters['num_topics'], alpha=self.hyperparameters['alpha'],
                          eta=self.hyperparameters['eta'])

        for i in self.x_train:
            lda.add_doc(i)

        lda.train(self.hyperparameters['max_iters'])

        topic_word_matrix = np.stack(
            [lda.get_topic_word_dist(k, normalize=True) for k in range(lda.k)])  # topic word distribution matrix
        topic_document_matrix = np.stack(
            [doc.get_topic_dist() for doc in lda.docs])  # topic document distribution matrix

        additional_words = [item for item in self.vocabulary if item not in list(lda.used_vocabs)]
        num_additional_words = len(additional_words)
        if num_additional_words > 0:
            topic_word_matrix = np.concatenate(
                (topic_word_matrix, np.zeros((topic_word_matrix.shape[0], num_additional_words), dtype=float)), axis=1)
        #new_topic_word_matrix = np.zeros(topic_word_matrix.shape)
        final_vocab = list(lda.used_vocabs) + additional_words
        vocab2id = {w: i for i, w in enumerate(final_vocab)}


        sorted_indexes = [vocab2id[w] for i, w in enumerate(self.vocabulary)]
        topic_word_matrix = topic_word_matrix[:, sorted_indexes]

        # topics extraction
        topic_w = []
        for k in range(lda.k):
            topics = []
            for word in lda.get_topic_words(k):
                topics.append(word[0])
            topic_w.append(topics)

        # Output model on the Train Set
        info = {}
        info['topics'] = topic_w
        info['topic-word-matrix'] = topic_word_matrix
        info['topic-document-matrix'] = topic_document_matrix.T

        # Inference on the test set
        if self.x_test is not None:
            doc_inst = [lda.make_doc(i) for i in self.x_test]
            topic_dist, _ = lda.infer(doc_inst)  # topic document distribution
            info['test-topic-document-matrix'] = np.asarray(topic_dist).T

        return info

    def partitioning(self, use_partitions=False):
        self.use_partitions = use_partitions

    def set_default_hyperparameters(self, hyperparameters):
        for k in hyperparameters.keys():
            if k in self.hyperparameters.keys():
                self.hyperparameters[k] = hyperparameters.get(k, self.hyperparameters[k])
