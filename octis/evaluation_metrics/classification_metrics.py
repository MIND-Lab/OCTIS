import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import octis.configuration.citations as citations
from octis.evaluation_metrics.metrics import AbstractMetric
from sklearn.preprocessing import MultiLabelBinarizer

stored_average = None
stored_use_log = None
stored_scale = None
stored_kernel = None
stored_model_output_hash = None
stored_svm_results = [None, None]


class ClassificationScore(AbstractMetric):
    def __init__(self, dataset, average='micro', use_log=False, scale=True, kernel='linear', same_svm=False):
        AbstractMetric.__init__(self)

        self._train_document_representations = None
        self._test_document_representations = None

        self._labels = dataset.get_labels()
        self.average = average

        self.same_svm = same_svm

        self.use_log = use_log
        self.scale = scale

        self.kernel = kernel

    def score(self, model_output):
        self._train_document_representations = model_output["topic-document-matrix"].T
        self._test_document_representations = model_output["test-topic-document-matrix"].T

        if self.use_log:
            self._train_document_representations = np.log(
                self._train_document_representations)
            self._test_document_representations = np.log(
                self._test_document_representations)
        if self.scale:
            # scaler = MinMaxScaler()
            scaler2 = StandardScaler()
            x_train = scaler2.fit_transform(
                self._train_document_representations)
            x_test = scaler2.transform(self._test_document_representations)
        else:
            x_train = self._train_document_representations
            x_test = self._test_document_representations

        train_labels = [label for label in self._labels[:len(x_train)]]
        test_labels = [label for label in self._labels[-len(x_test):]]

        if type(self._labels[0]) == list:
            mlb = MultiLabelBinarizer()

            train_labels = mlb.fit_transform(train_labels)
            test_labels = mlb.transform(test_labels)
            clf = RandomForestClassifier()

        else:
            label2id = {}
            for i, lab in enumerate(list(train_labels)):
                label2id[lab] = i

            train_labels = [label2id[l] for l in train_labels]
            test_labels = [label2id[l] for l in test_labels]

            if self.kernel == 'linear':
                clf = svm.LinearSVC(verbose=False)
            else:
                clf = svm.SVC(kernel=self.kernel, verbose=False)

        ###########
        clf.fit(x_train, train_labels)

        predicted_test_labels = clf.predict(x_test)

        return test_labels, predicted_test_labels


def compute_SVM_output(model_output, metric, super_metric):
    global stored_average
    global stored_use_log
    global stored_scale
    global stored_kernel
    global stored_svm_results
    global stored_model_output_hash

    model_output_hash = hash(str(model_output))

    test_labels = None
    predicted_test_labels = None
    flag = True

    if stored_average == metric.average and \
        stored_use_log == metric.use_log and \
        stored_scale == metric.scale and \
        stored_kernel == metric.kernel and \
        stored_model_output_hash == model_output_hash:
        test_labels, predicted_test_labels = stored_svm_results
    else:
        test_labels, predicted_test_labels = super_metric.score(model_output)

        stored_average = metric.average
        stored_use_log = metric.use_log
        stored_scale = metric.scale
        stored_kernel = metric.kernel
        stored_svm_results = [test_labels, predicted_test_labels]
        stored_model_output_hash = model_output_hash
        flag = False

    return [test_labels, predicted_test_labels, flag]


class F1Score(ClassificationScore):
    def __init__(self, dataset, average='micro', use_log=False, scale=True, kernel='linear', same_svm=False):
        super().__init__(dataset=dataset, average=average,
                         use_log=use_log, scale=scale, kernel=kernel, same_svm=same_svm)

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
        test_labels, predicted_test_labels, self.same_svm = compute_SVM_output(model_output, self, super())
        return f1_score(test_labels, predicted_test_labels, average=self.average)


class PrecisionScore(ClassificationScore):

    def __init__(self, dataset, average='micro', use_log=False, scale=True, kernel='linear', same_svm=False):
        super().__init__(dataset=dataset, average=average,
                         use_log=use_log, scale=scale, kernel=kernel, same_svm=same_svm)

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
        test_labels, predicted_test_labels, self.same_svm = compute_SVM_output(model_output, self, super())
        return precision_score(test_labels, predicted_test_labels, average=self.average)


class RecallScore(ClassificationScore):

    def __init__(self, dataset, average='micro', use_log=False, scale=True, kernel='linear', same_svm=False):
        super().__init__(dataset=dataset, average=average,
                         use_log=use_log, scale=scale, kernel=kernel, same_svm=same_svm)

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
        test_labels, predicted_test_labels, self.same_svm = compute_SVM_output(model_output, self, super())
        return recall_score(test_labels, predicted_test_labels, average=self.average)


class AccuracyScore(ClassificationScore):

    def __init__(self, dataset, average='micro', use_log=False, scale=True, kernel='linear', same_svm=False):
        super().__init__(dataset=dataset, average=average,
                         use_log=use_log, scale=scale, kernel=kernel, same_svm=same_svm)

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
        test_labels, predicted_test_labels, self.same_svm = compute_SVM_output(model_output, self, super())
        return accuracy_score(test_labels, predicted_test_labels)
