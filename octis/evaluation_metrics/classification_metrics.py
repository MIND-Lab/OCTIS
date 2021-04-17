from octis.evaluation_metrics.metrics import AbstractMetric
import octis.configuration.citations as citations
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from libsvm.svmutil import *


class ClassificationScore(AbstractMetric):
    def __init__(self, dataset, average='micro', use_log=False, scale=True, kernel='linear'):
        AbstractMetric.__init__(self)

        self._train_document_representations = None
        self._test_document_representations = None

        self._labels = dataset.get_labels()
        self.average = average

        self.use_log = use_log
        self.scale = scale

        self.kernel = kernel

    def score(self, model_output):
        self._train_document_representations = model_output["topic-document-matrix"].T
        self._test_document_representations = model_output["test-topic-document-matrix"].T

        if self.use_log:
            self._train_document_representations = np.log(self._train_document_representations)
            self._test_document_representations = np.log(self._test_document_representations)
        if self.scale:
            #scaler = MinMaxScaler()
            scaler2 = StandardScaler()
            x_train = scaler2.fit_transform(self._train_document_representations)
            x_test = scaler2.transform(self._test_document_representations)
        else:
            x_train = self._train_document_representations
            x_test = self._test_document_representations
        train_labels = [label for label in self._labels[:len(x_train)]]
        test_labels = [label for label in self._labels[-len(x_test):]]

        id2label, label2id = {}, {}
        for i, lab in enumerate(list(train_labels)):
            id2label[i] = lab
            label2id[lab] = i

        train_labels = [label2id[l] for l in train_labels]
        test_labels = [label2id[l] for l in test_labels]

        '''
        m = svm_train(train_labels, X_train, '-t 0')# -S 0 -K 2 -Z ')
        p_label, _, _ = svm_predict(test_labels, X_test, m)
        #print(len(X_test))
        #print(X_test.shape)
        return f1_score(test_labels, p_label, average=self.average)
        '''

        if self.kernel == 'linear':
            clf = svm.LinearSVC(verbose=False)
        else:
            clf = svm.SVC(kernel=self.kernel, verbose=False)
        clf.fit(x_train, train_labels)

        predicted_test_labels = clf.predict(x_test)

        return test_labels, predicted_test_labels


class F1Score(ClassificationScore):
    def __init__(self, dataset, average='micro', use_log=False, scale=True, kernel='linear'):
        super().__init__(dataset=dataset, average=average, use_log=use_log, scale=scale, kernel=kernel)

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
        model_output : dictionary, output of the model. keys 'topic-document-matrix' and
                       'test-topic-document-matrix' are required.

        Returns
        -------
        score : score
        """

        test_labels, predicted_test_labels = super().score(model_output)
        return f1_score(test_labels, predicted_test_labels, average=self.average)


class PrecisionScore(ClassificationScore):

    def __init__(self, dataset, average='micro', use_log=False, scale=True, kernel='linear'):
        super().__init__(dataset=dataset, average=average, use_log=use_log, scale=scale, kernel=kernel)

    def info(self):
        return {"citation": citations.em_f1_score, "name": "Precision"}

    def score(self, model_output):
        """
        Retrieves the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model. 'topic-document-matrix' and
                       'test-topic-document-matrix' keys are required.

        Returns
        -------
        score : score
        """
        test_labels, predicted_test_labels = super().score(model_output)
        return precision_score(test_labels, predicted_test_labels, average=self.average)


class RecallScore(ClassificationScore):

    def __init__(self, dataset, average='micro', use_log=False, scale=True, kernel='linear'):
        super().__init__(dataset=dataset, average=average, use_log=use_log, scale=scale, kernel=kernel)

    def info(self):
        return {"citation": citations.em_f1_score, "name": "Precision"}

    def score(self, model_output):
        """
        Retrieves the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model. 'topic-document-matrix' and
                       'test-topic-document-matrix' keys are required.

        Returns
        -------
        score : score
        """
        test_labels, predicted_test_labels = super().score(model_output)
        return recall_score(test_labels, predicted_test_labels, average=self.average)
