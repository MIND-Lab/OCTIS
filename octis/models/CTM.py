from sklearn.feature_extraction.text import CountVectorizer

from octis.models.model import AbstractModel
from octis.models.contextualized_topic_models.datasets import dataset
from octis.models.contextualized_topic_models.models import ctm
from octis.models.contextualized_topic_models.utils.data_preparation import (
    bert_embeddings_from_list)

import os
import pickle as pkl
import torch
import numpy as np
import random


class CTM(AbstractModel):

    def __init__(
        self, num_topics=10, model_type='prodLDA', activation='softplus',
        dropout=0.2, learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99,
        solver='adam', num_epochs=100, reduce_on_plateau=False, prior_mean=0.0,
        prior_variance=None, num_layers=2, num_neurons=100, seed=None,
        use_partitions=True, num_samples=10, inference_type="zeroshot",
            bert_path="", bert_model="bert-base-nli-mean-tokens"):
        """
        initialization of CTM

        :param num_topics : int, number of topic components, (default 10)
        :param model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
        :param activation : string, 'softplus', 'relu', 'sigmoid', 'swish',
            'tanh', 'leakyrelu', 'rrelu', 'elu', 'selu' (default 'softplus')
        :param num_layers : int, number of layers (default 2)
        :param dropout : float, dropout to use (default 0.2)
        :param learn_priors : bool, make priors a learnable parameter
            (default True)
        :param batch_size : int, size of batch to use for training (default 64)
        :param lr : float, learning rate to use for training (default 2e-3)
        :param momentum : float, momentum to use for training (default 0.99)
        :param solver : string, optimizer 'adam' or 'sgd' (default 'adam')
        :param num_epochs : int, number of epochs to train for, (default 100)
        :param num_samples: int, number of times theta needs to be sampled
            (default: 10)
        :param seed : int, the random seed. Not used if None (default None).
        :param use_partitions: bool, if true the model will be trained on the
            training set and evaluated on the test set (default: true)
        :param reduce_on_plateau : bool, reduce learning rate by 10x on
            plateau of 10 epochs (default False)
        :param inference_type: the type of the CTM model. It can be "zeroshot"
            or "combined" (default zeroshot)
        :param bert_path: path to store the document contextualized
            representations
        :param bert_model: name of the contextualized model
            (default: bert-base-nli-mean-tokens).
            see https://www.sbert.net/docs/pretrained_models.html
        """

        super().__init__()

        self.hyperparameters['num_topics'] = num_topics
        self.hyperparameters['model_type'] = model_type
        self.hyperparameters['activation'] = activation
        self.hyperparameters['dropout'] = dropout
        self.hyperparameters['inference_type'] = inference_type
        self.hyperparameters['learn_priors'] = learn_priors
        self.hyperparameters['batch_size'] = batch_size
        self.hyperparameters['lr'] = lr
        self.hyperparameters['num_samples'] = num_samples
        self.hyperparameters['momentum'] = momentum
        self.hyperparameters['solver'] = solver
        self.hyperparameters['num_epochs'] = num_epochs
        self.hyperparameters['reduce_on_plateau'] = reduce_on_plateau
        self.hyperparameters["prior_mean"] = prior_mean
        self.hyperparameters["prior_variance"] = prior_variance
        self.hyperparameters["num_neurons"] = num_neurons
        self.hyperparameters["bert_path"] = bert_path
        self.hyperparameters["num_layers"] = num_layers
        self.hyperparameters["bert_model"] = bert_model
        self.hyperparameters["seed"] = seed
        self.use_partitions = use_partitions

        hidden_sizes = tuple([num_neurons for _ in range(num_layers)])
        self.hyperparameters['hidden_sizes'] = tuple(hidden_sizes)

        self.model = None
        self.vocab = None

    def train_model(self, dataset, hyperparameters=None, top_words=10):
        """
        trains CTM model

        :param dataset: octis Dataset for training the model
        :param hyperparameters: dict, with optionally) the following information:
        :param top_words: number of top-n words of the topics (default 10)

        """
        if hyperparameters is None:
            hyperparameters = {}

        self.set_params(hyperparameters)
        self.vocab = dataset.get_vocabulary()
        self.set_seed(seed=self.hyperparameters['seed'])

        if self.use_partitions:
            train, validation, test = dataset.get_partitioned_corpus(
                use_validation=True)

            data_corpus_train = [' '.join(i) for i in train]
            data_corpus_test = [' '.join(i) for i in test]
            data_corpus_validation = [' '.join(i) for i in validation]

            x_train, x_test, x_valid, input_size = self.preprocess(
                self.vocab, data_corpus_train, test=data_corpus_test,
                validation=data_corpus_validation,
                bert_train_path=(
                    self.hyperparameters['bert_path'] + "_train.pkl"),
                bert_test_path=self.hyperparameters['bert_path'] + "_test.pkl",
                bert_val_path=self.hyperparameters['bert_path'] + "_val.pkl",
                bert_model=self.hyperparameters["bert_model"])

            self.model = ctm.CTM(
                input_size=input_size, bert_input_size=x_train.X_bert.shape[1], model_type='prodLDA',
                num_topics=self.hyperparameters['num_topics'], dropout=self.hyperparameters['dropout'],
                activation=self.hyperparameters['activation'], lr=self.hyperparameters['lr'],
                inference_type=self.hyperparameters['inference_type'],
                hidden_sizes=self.hyperparameters['hidden_sizes'],
                solver=self.hyperparameters['solver'],
                momentum=self.hyperparameters['momentum'],
                num_epochs=self.hyperparameters['num_epochs'],
                learn_priors=self.hyperparameters['learn_priors'],
                batch_size=self.hyperparameters['batch_size'],
                num_samples=self.hyperparameters['num_samples'],
                topic_prior_mean=self.hyperparameters["prior_mean"],
                reduce_on_plateau=self.hyperparameters['reduce_on_plateau'],
                topic_prior_variance=self.hyperparameters["prior_variance"],
                top_words=top_words)

            self.model.fit(x_train, x_valid, verbose=self.hyperparameters["verbose"], save_dir=self.hyperparameters["save_dir"])
            result = self.inference(x_test)
            return result

        else:
            data_corpus = [' '.join(i) for i in dataset.get_corpus()]
            x_train, input_size = self.preprocess(
                self.vocab, train=data_corpus,
                bert_train_path=(
                    self.hyperparameters['bert_path'] + "_train.pkl"),
                bert_model=self.hyperparameters["bert_model"])

        self.model = ctm.CTM(
             input_size=input_size, bert_input_size=x_train.X_bert.shape[1], model_type='prodLDA',
             num_topics=self.hyperparameters['num_topics'], dropout=self.hyperparameters['dropout'],
             activation=self.hyperparameters['activation'], lr=self.hyperparameters['lr'],
             inference_type=self.hyperparameters['inference_type'],
             hidden_sizes=self.hyperparameters['hidden_sizes'], solver=self.hyperparameters['solver'],
             momentum=self.hyperparameters['momentum'], num_epochs=self.hyperparameters['num_epochs'],
             learn_priors=self.hyperparameters['learn_priors'],
             batch_size=self.hyperparameters['batch_size'],
             num_samples=self.hyperparameters['num_samples'],
             topic_prior_mean=self.hyperparameters["prior_mean"],
             reduce_on_plateau=self.hyperparameters['reduce_on_plateau'],
             topic_prior_variance=self.hyperparameters["prior_variance"],
             top_words=top_words)

        self.model.fit(x_train, None, verbose=self.hyperparameters["verbose"], save_dir=self.hyperparameters["save_dir"])
        result = self.model.get_info()
        return result

    def set_params(self, hyperparameters):
        for k in hyperparameters.keys():
            if k in self.hyperparameters.keys() and k != 'hidden_sizes':
                self.hyperparameters[k] = hyperparameters.get(
                    k, self.hyperparameters[k])

        self.hyperparameters['hidden_sizes'] = tuple(
            [self.hyperparameters["num_neurons"] for _ in range(
                self.hyperparameters["num_layers"])])

    def inference(self, x_test):
        assert isinstance(self.use_partitions, bool) and self.use_partitions
        results = self.model.predict(x_test)
        return results

    def partitioning(self, use_partitions=False):
        self.use_partitions = use_partitions

    @staticmethod
    def set_seed(seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.deterministic = True

    @staticmethod
    def preprocess(
        vocab, train, bert_model, test=None, validation=None,
            bert_train_path=None, bert_test_path=None, bert_val_path=None):
        vocab2id = {w: i for i, w in enumerate(vocab)}
        vec = CountVectorizer(
            vocabulary=vocab2id, token_pattern=r'(?u)\b[\w+|\-]+\b')
        entire_dataset = train.copy()
        if test is not None:
            entire_dataset.extend(test)
        if validation is not None:
            entire_dataset.extend(validation)

        vec.fit(entire_dataset)
        idx2token = {v: k for (k, v) in vec.vocabulary_.items()}

        x_train = vec.transform(train)
        b_train = CTM.load_bert_data(bert_train_path, train, bert_model)

        train_data = dataset.CTMDataset(x_train.toarray(), b_train, idx2token)
        input_size = len(idx2token.keys())

        if test is not None and validation is not None:
            x_test = vec.transform(test)
            b_test = CTM.load_bert_data(bert_test_path, test, bert_model)
            test_data = dataset.CTMDataset(x_test.toarray(), b_test, idx2token)

            x_valid = vec.transform(validation)
            b_val = CTM.load_bert_data(bert_val_path, validation, bert_model)
            valid_data = dataset.CTMDataset(
                x_valid.toarray(), b_val, idx2token)
            return train_data, test_data, valid_data, input_size
        if test is None and validation is not None:
            x_valid = vec.transform(validation)
            b_val = CTM.load_bert_data(bert_val_path, validation, bert_model)
            valid_data = dataset.CTMDataset(
                x_valid.toarray(), b_val, idx2token)
            return train_data, valid_data, input_size
        if test is not None and validation is None:
            x_test = vec.transform(test)
            b_test = CTM.load_bert_data(bert_test_path, test, bert_model)
            test_data = dataset.CTMDataset(x_test.toarray(), b_test, idx2token)
            return train_data, test_data, input_size
        if test is None and validation is None:
            return train_data, input_size

    @staticmethod
    def load_bert_data(bert_path, texts, bert_model):
        if bert_path is not None:
            if os.path.exists(bert_path):
                bert_ouput = pkl.load(open(bert_path, 'rb'))
            else:
                bert_ouput = bert_embeddings_from_list(texts, bert_model)
                pkl.dump(bert_ouput, open(bert_path, 'wb'))
        else:
            bert_ouput = bert_embeddings_from_list(texts, bert_model)
        return bert_ouput
