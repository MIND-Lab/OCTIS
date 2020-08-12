from models.model import Abstract_Model
import pandas as pd
import numpy as np
import tomotopy as tp

class FastLDA(Abstract_Model):

    def __init__(self):
        self.hyperparameters = {}

    def train_model(self, dataset, hyperparameters, topic_word_matrix=True, topic_document_matrix=True):

        if self.use_partitions:
            X_train, X_test = dataset.get_partitioned_corpus()
        else:
            X_train = dataset.get_corpus()
            X_test = None

        mdl = tp.LDAModel(k=self.hyperparameters['num_topics'])

        for i in X_train:
            mdl.add_doc(i)

        mdl.train(self.hyperparameters['max_iter'])

        topic_word_matrix = np.stack(
            [mdl.get_topic_word_dist(k) for k in range(mdl.k)])  # topic word distribution matrix
        topic_document_matrix = np.stack([doc.get_topic_dist() for doc in mdl.docs])  # topic document distribution matrix

        # topics extraction
        topic_w = []
        for k in range(mdl.k):
            topics = []
            for word in mdl.get_topic_words(k):
                topics.append(word[0])
            topic_w.append(topics)

        info = {}
        info['topics'] = np.asarray(topic_w)
        info['topic-word-matrix'] = topic_word_matrix
        info['topic-document-matrix'] = topic_document_matrix

        if X_test is not None:
            doc_inst = [mdl.make_doc(i) for i in X_test]
            topic_dist, _ = mdl.infer(doc_inst)  # topic document distribution
            info['test-topic-document-matrix'] = np.asarray(topic_dist)

        return info

    def partitioning(self, use_partitions=False):

        if use_partitions:
            self.use_partitions = True
        else:
            self.use_partitions = False

    def set_default_hyperparameters(self, hyperparameters):
        self.hyperparameters['num_topics'] = hyperparameters.get(
            'num_topics', self.hyperparameters.get('num_topics', 10))
        self.hyperparameters['max_iter'] = hyperparameters.get(
            'model_type', self.hyperparameters.get('max_iter', 50))
