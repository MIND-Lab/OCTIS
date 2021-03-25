from octis.evaluation_metrics.metrics import Abstract_Metric
import octis.configuration.citations as citations
import octis.configuration.defaults as defaults
from sklearn.metrics import f1_score
import numpy as np
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from libsvm.svmutil import *


class F1Score(Abstract_Metric):

    def __init__(self, metric_parameters=None):
        Abstract_Metric.__init__(self, metric_parameters)
        if metric_parameters is None:
            metric_parameters = {}
        self.train_document_representations = None
        self.test_document_representations = None
        parameters = defaults.em_f1_score.copy()
        parameters.update(metric_parameters)
        self.parameters = parameters
        if 'dataset' not in metric_parameters.keys():
            raise Exception('A dataset is required to extract the labels')
        else:
            self.labels = metric_parameters['dataset'].get_labels()
        self.average = parameters['average']

        if 'use_log' in metric_parameters:
            self.use_log = metric_parameters['use_log']
        else:
            self.use_log = False
        if 'scale' in metric_parameters:
            self.scale = metric_parameters['scale']
        else:
            self.scale = True

        if 'kernel' in metric_parameters:
            self.kernel = metric_parameters['kernel']
        else:
            self.kernel = 'linear'

    def info(self):
        return {
            "citation": citations.em_f1_score,
            "name": "F1 Score"
        }

    def score(self, model_output):
        """
        Retrieves the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       key 'topic-document-matrix' and
                       'test-topic-document-matrix' required.

        Returns
        -------
        score : score
        """

        self.train_document_representations = model_output["topic-document-matrix"].T
        self.test_document_representations = model_output["test-topic-document-matrix"].T

        if self.use_log:
            self.train_document_representations = np.log(self.train_document_representations)
            self.test_document_representations = np.log(self.test_document_representations)
        if self.scale:
            #scaler = MinMaxScaler()
            scaler2 = StandardScaler()
            X_train = scaler2.fit_transform(self.train_document_representations)
            X_test = scaler2.transform(self.test_document_representations)
        else:
            X_train = self.train_document_representations
            X_test = self.test_document_representations
        train_labels = [l for l in self.labels[:len(X_train)]]
        test_labels = [l for l in self.labels[-len(X_test):]]

        id2label = {}
        label2id = {}
        count = 0
        for i in set(train_labels):
            id2label[count] = i
            label2id[i] = count
            count = count + 1

        train_labels = [label2id[l] for l in train_labels]
        test_labels = [label2id[l] for l in test_labels]

        if self.kernel == 'linear':
            clf = svm.LinearSVC(verbose=True)
        else:
            clf = svm.SVC(kernel=self.kernel)
        clf.fit(X_train, train_labels)

        predicted_test_labels = clf.predict(X_test)

        return f1_score(test_labels, predicted_test_labels, average=self.average)

        '''
        m = svm_train(train_labels, X_train, '-t 0')# -S 0 -K 2 -Z ')
        p_label, _, _ = svm_predict(test_labels, X_test, m)
        #print(len(X_test))
        #print(X_test.shape)
        return f1_score(test_labels, p_label, average=self.average)
        '''

