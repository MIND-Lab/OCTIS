
from models.model import Abstract_Model
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 
import numpy as np 

from models.pytorchavitm.avitm import avitm
from models.pytorchavitm import datasets


class TorchAvitm(Abstract_Model):

    def __init__(self):
        self.hyperparameters={}

    def train_model(self, dataset, hyperparameters, top_words=10, topic_word_matrix=True, topic_document_matrix=True):
        """
            Args
                dataset: list of sentences for training the model
                hyparameters: dict, with the below information:

                input_size : int, dimension of input
                n_components : int, number of topic components, (default 10)
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


        if self.test == True:
            data = dataset.get_partitioned_corpus()
            X_train = data[0]
            X_test = data[1]
            data_corpus_train = [','.join(i) for i in X_train]
            data_corpus_test = [','.join(i) for i in X_test]
            X_train, self.X_test, input_size = self.preprocess(data_corpus_train, data_corpus_test)
        else:
            data_corpus = [','.join(i) for i in dataset.get_corpus()]
            X_train, input_size = self.preprocess(data_corpus)
      
        self.avitm_model = avitm.AVITM(input_size=input_size,
                                  n_components=self.hyperparameters['n_components'],
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
                                      "prior_variance"])
    
        self.avitm_model.fit(X_train)
        result = self.avitm_model.get_info()
        
        return result

    def inference(self):
        assert isinstance(self.test, bool) and self.test == True

        results = self.avitm_model.predict(self.X_test)

        return results

    def set_default_hyperparameters(self, hyperparameters):
        self.hyperparameters['num_topics'] = hyperparameters.get(
            'num_topics', self.hyperparameters.get('num_topics', 10))
        self.hyperparameters['model_type'] = hyperparameters.get(
            'model_type',self.hyperparameters.get('model_type', 'prodLDA'))
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
            'prior_variance', self.hyperparameters.get('prior_variance', 0.0))

        default_hidden_sizes = [100, 100, 0]
        hidden_sizes = [hyperparameters.get(
            'layer_' + str(0), self.hyperparameters.get(
                'layer_' + str(0), default_hidden_sizes[0]))]
        for i in range(1, 3):
            curr_layer = hyperparameters.get(
                'layer_' + str(i), self.hyperparameters.get(
                    'layer_' + str(i), default_hidden_sizes[i]))
            if curr_layer > 0:
                hidden_sizes.append(curr_layer)
            else:
                break

        self.hyperparameters['hidden_sizes'] = tuple(hidden_sizes)

    def test_set(self, test_input=False):
        if test_input:
            self.test = True
        else:
            self.test = False

    @staticmethod
    def preprocess(data, test = None):
        
        def to_bow(data, min_length):
            """Convert index lists to bag of words representation of documents."""
            vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length)
                    for x in data if np.sum(x[x != np.array(None)]) != 0]
            return np.array(vect)
        if test is not None:
            vec = CountVectorizer()
            X_train = vec.fit_transform(data)
            X_test = vec.transform(test)
            idx2token = {v: k for (k, v) in vec.vocabulary_.items()}
            #train_bow = to_bow(X_train.toarray(), len(idx2token.keys()))
            #test_bow = to_bow(X_test.toarray(), len(idx2token.keys()))
            train_data = datasets.BOWDataset(X_train, idx2token)
            test_data = datasets.BOWDataset(X_test, idx2token)
            input_size = len(idx2token.keys())

            return train_data,test_data, input_size

        else:
            vec = CountVectorizer()
            X = vec.fit_transform(data)
            idx2token = {v: k for (k, v) in vec.vocabulary_.items()}
            #train_bow = to_bow(X.toarray(), len(idx2token.keys()))
            train_data = datasets.BOWDataset(X, idx2token)
            input_size = len(idx2token.keys())
            return train_data, input_size
    



