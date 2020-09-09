
from models.model import Abstract_Model
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from models.pytorchavitm.avitm import avitm
from models.pytorchavitm import datasets


class TorchAvitm(Abstract_Model):

    def __init__(self):
        self.hyperparameters={}

    def train_model(self, dataset, hyperparameters, top_words=10,
                    topic_word_matrix=True, topic_document_matrix=True):
        """
            Args
                dataset: list of sentences for training the model
                hyparameters: dict, with the below information:

                input_size : int, dimension of input
                num_topics : int, number of topic components, (default 10)
                model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
                hidden_sizes : tuple, length = n_layers, (default (100, 100))
                activation : string, 'softplus', 'relu', (default 'softplus')
                dropout : float, dropout to use (default 0.2)
                learn_priors : bool, make priors a learnable parameter (default True)
                batch_size : int, size of batch to use for training (default 64)
                lr : float, learning rate to use for training (default 2e-3)
                momentum : float, momentum to use for training (default 0.99)
                solver : string, optimizer 'adam' or 'sgd' (default 'adam')
                num_epochs : int, number of epochs to train for, (default 100)
                reduce_on_plateau : bool, reduce learning rate by 10x on plateau of 10 epochs (default False)
            """

        self.set_default_hyperparameters(hyperparameters)
        self.bool_topic_doc = topic_document_matrix
        self.bool_topic_word = topic_word_matrix

        if self.use_partitions:
            train, validation, test = dataset.get_partitioned_corpus(use_validation=True)

            data_corpus_train = [' '.join(i) for i in train]
            data_corpus_test = [' '.join(i) for i in test]
            data_corpus_validation = [' '.join(i) for i in validation]

            self.vocab = dataset.get_vocabulary()
            self.X_train, self.X_test, self.X_valid, input_size = \
                self.preprocess(self.vocab, data_corpus_train, test=data_corpus_test,
                                validation=data_corpus_validation)
        else:
            data_corpus = [' '.join(i) for i in dataset.get_corpus()]
            self.X_train, input_size = self.preprocess(self.vocab, train=data_corpus)
      
        self.model = avitm.AVITM(input_size=input_size,
                                 num_topics=self.hyperparameters['num_topics'],
                                 model_type=self.hyperparameters['model_type'],
                                 hidden_sizes=self.hyperparameters['hidden_sizes'],
                                 activation=self.hyperparameters['activation'],
                                 dropout=self.hyperparameters['dropout'],
                                 learn_priors=self.hyperparameters['learn_priors'],
                                 batch_size=self.hyperparameters['batch_size'],
                                 lr=self.hyperparameters['lr'],
                                 momentum=self.hyperparameters['momentum'],
                                 solver=self.hyperparameters['solver'],
                                 num_epochs=self.hyperparameters['num_epochs'],
                                 reduce_on_plateau=self.hyperparameters[
                                           'reduce_on_plateau'],
                                 topic_prior_mean=self.hyperparameters["prior_mean"],
                                 topic_prior_variance=self.hyperparameters[
                                           "prior_variance"],
                                 topic_word_matrix=self.bool_topic_word,
                                 topic_document_matrix=self.bool_topic_doc
                                 )
    
        self.model.fit(self.X_train, self.X_valid)
        
        if self.use_partitions:
            result = self.inference()
        else:
            result = self.model.get_info()
        return result

    def inference(self):
        assert isinstance(self.use_partitions, bool) and self.use_partitions
        results = self.model.predict(self.X_test)
        return results

    def set_default_hyperparameters(self, hyperparameters):
        self.hyperparameters['num_topics'] = hyperparameters.get(
            'num_topics', self.hyperparameters.get('num_topics', 10))
        self.hyperparameters['model_type'] = hyperparameters.get(
            'model_type', self.hyperparameters.get('model_type', 'prodLDA'))
        self.hyperparameters['activation'] = hyperparameters.get(
            'activation', self.hyperparameters.get('activation', 'softplus'))
        self.hyperparameters['dropout'] = hyperparameters.get(
            'dropout', self.hyperparameters.get('dropout', 0.2))
        self.hyperparameters['learn_priors'] = hyperparameters.get(
            'learn_priors', self.hyperparameters.get('learn_priors', True))
        self.hyperparameters['batch_size'] = hyperparameters.get(
            'batch_size', self.hyperparameters.get('batch_size', 64))
        self.hyperparameters['lr'] = hyperparameters.get(
            'lr', self.hyperparameters.get('lr', 2e-3))
        self.hyperparameters['momentum'] = hyperparameters.get(
            'momentum', self.hyperparameters.get('momentum', 0.99))
        self.hyperparameters['solver'] = hyperparameters.get(
            'solver', self.hyperparameters.get('solver', 'adam'))
        self.hyperparameters['num_epochs'] = hyperparameters.get(
            'num_epochs', self.hyperparameters.get('num_epochs', 100))
        self.hyperparameters['reduce_on_plateau'] = hyperparameters.get(
            'reduce_on_plateau', self.hyperparameters.get('reduce_on_plateau', False))
        self.hyperparameters["prior_mean"] = hyperparameters.get(
            'prior_mean', self.hyperparameters.get('prior_mean', 0.0))
        self.hyperparameters["prior_variance"] = hyperparameters.get(
            'prior_variance', self.hyperparameters.get('prior_variance', None))

        default_hidden_sizes = [100, 100, 0, 0, 0]
        hidden_sizes = [hyperparameters.get(
            'layer_' + str(0), self.hyperparameters.get(
                'layer_' + str(0), default_hidden_sizes[0]))]
        for i in range(1, 5):
            curr_layer = hyperparameters.get(
                'layer_' + str(i), self.hyperparameters.get(
                    'layer_' + str(i), default_hidden_sizes[i]))
            if curr_layer > 0:
                hidden_sizes.append(curr_layer)
            else:
                break

        self.hyperparameters['hidden_sizes'] = tuple(hidden_sizes)

    def partitioning(self, use_partitions=False):
        self.use_partitions = use_partitions

    def info_test(self):
        if self.use_partitions:
            return self.X_test
        else:
            print('No partitioned dataset, please apply test_set method = True')

    @staticmethod
    def preprocess(vocab, train, test=None, validation=None):
        vocab2id = {w: i for i, w in enumerate(vocab)}
        vec = CountVectorizer(
            vocabulary=vocab2id, token_pattern=r'(?u)\b\w+\b')
        entire_dataset = train.copy()
        if test is not None:
            entire_dataset.extend(test)
        if validation is not None:
            entire_dataset.extend(validation)

        vec.fit(entire_dataset)
        idx2token = {v: k for (k, v) in vec.vocabulary_.items()}
        X_train = vec.transform(train)
        train_data = datasets.BOWDataset(X_train.toarray(), idx2token)
        input_size = len(idx2token.keys())

        if test is not None and validation is not None:
            X_test = vec.transform(test)
            test_data = datasets.BOWDataset(X_test.toarray(), idx2token)
            X_valid = vec.transform(validation)
            valid_data = datasets.BOWDataset(X_valid.toarray(), idx2token)
            return train_data, test_data, valid_data, input_size
        if test is None and validation is not None:
            X_valid = vec.transform(validation)
            valid_data = datasets.BOWDataset(X_valid.toarray(), idx2token)
            return train_data, valid_data, input_size
        if test is not None and validation is None:
            X_test = vec.transform(test)
            test_data = datasets.BOWDataset(X_test.toarray(), idx2token)
            return train_data, test_data, input_size
        if test is None and validation is None:
            return train_data, input_size



