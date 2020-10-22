from optopic.models.model import Abstract_Model
import pandas as pd
import numpy as np
import tomotopy as tp

class FastLDA(Abstract_Model):

    def __init__(self):
        self.hyperparameters = {}

    def train_model(self, dataset, hyperparameters, topic_word_matrix=True, topic_document_matrix=True):
        
        if self.use_partitions:
            X_train, X_test = dataset.get_partitioned_corpus(use_validation=False)
        else:
            X_train = dataset.get_corpus()
            X_test = None

        mdl = tp.LDAModel(k=self.hyperparameters['num_topics'], alpha = self.hyperparameters['alpha'],
                         eta = self.hyperparameters['eta'], min_cf = self.hyperparameters['min_cf'],
                         min_df = self.hyperparameters['min_df'], rm_top = self.hyperparameters['rm_top'])
        
        for i in X_train:
            mdl.add_doc(i)

        mdl.train(self.hyperparameters['max_iter'])
        
        bool_topic_word = topic_word_matrix
        bool_topic_docu = topic_document_matrix
        
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
            
        #Output model on the Train Set
        info = {}
        info['topics'] = np.asarray(topic_w)
        info['topic-word-matrix'] = topic_word_matrix
        info['topic-document-matrix'] = topic_document_matrix
        
        #Inference on the test set 
        if X_test is not None:
            doc_inst = [mdl.make_doc(i) for i in X_test]
            topic_dist, _ = mdl.infer(doc_inst)  # topic document distribution
            info['test-topic-document-matrix'] = np.asarray(topic_dist)
            
        #Manage the model output
        if bool_topic_word and bool_topic_docu:
            info_diz = info
        elif bool_topic_word and not bool_topic_docu:
            info_diz = {}
            info_diz['topics'] = info['topics']
            info_diz['topic-word-matrix'] = info['topic-word-matrix'] 
        elif not bool_topic_word and bool_topic_docu:
            info_diz = {}
            info_diz['topics'] = info['topics']
            info_diz['topic-document-matrix'] = info['topic-document-matrix']
            if X_test is not None:
                info_diz['test-topic-document-matrix'] = info['test-topic-document-matrix'] 
        else:
            info_diz = {}
            info_diz['topics'] = info['topics']
        
        return info_diz

    def partitioning(self, use_partitions=False):
        if use_partitions:
            self.use_partitions = True
        else:
            self.use_partitions = False

    def set_default_hyperparameters(self, hyperparameters):
        self.hyperparameters['num_topics'] = hyperparameters.get(
            'num_topics', self.hyperparameters.get('num_topics', 10))
        self.hyperparameters['max_iter'] = hyperparameters.get(
            'max_iter', self.hyperparameters.get('max_iter', 50))
        self.hyperparameters['eta'] = hyperparameters.get('eta', 
            self.hyperparameters.get('eta', .01))
        self.hyperparameters['alpha'] = hyperparameters.get('alpha',
            self.hyperparameters.get('alpha', .1))
        self.hyperparameters['min_cf'] = hyperparameters.get('min_cf',
            self.hyperparameters.get('min_cf', 0))
        self.hyperparameters['min_df'] = hyperparameters.get('min_df',
            self.hyperparameters.get('min_df', 0))
        self.hyperparameters['rm_top'] = hyperparameters.get('rm_top',
            self.hyperparameters.get('rm_top', 0))
        
        
                                                          
                                                          
                                                          
                                                          
                                                          
                                                          
