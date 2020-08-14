from evaluation_metrics.metrics import Abstract_Metric
import configuration.citations as citations
import configuration.defaults as defaults
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

from sklearn import svm

class F1Score(Abstract_Metric):

    def __init__(self, metric_parameters={}):
        Abstract_Metric.__init__(self, metric_parameters)
        self.train_document_representations = None
        self.test_document_representations = None
        parameters = defaults.em_f1_score.copy()
        parameters.update(metric_parameters)
        if 'dataset' not in metric_parameters.keys():
            raise Exception('A dataset is required to extract the labels')
        else:
            self.labels = metric_parameters['dataset'].get_labels()
        self.average = parameters['average']
        if 'use_log' in metric_parameters:
            self.use_log = metric_parameters['use_log']
        else:
            self.use_log = False

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


        train_labels = [l[0] for l in self.labels[:len(self.train_document_representations)]]
        test_labels = [l[0] for l in self.labels[
                                     -len(self.test_documents_representations):]]

        clf = svm.SVC(kernel='poly')
        clf.fit(self.train_document_representations, train_labels)
        predicted_test_labels = clf.predict(self.test_documents_representations)

        return f1_score(test_labels, predicted_test_labels, average=self.average)
        #, confusion_matrix(test_labels, predicted_test_labels)
