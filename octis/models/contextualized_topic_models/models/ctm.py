import datetime
import os
from collections import defaultdict

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from octis.models.contextualized_topic_models.networks.decoding_network import DecoderNetwork
from octis.models.early_stopping.pytorchtools import EarlyStopping


class CTM(object):
    """Class to train the contextualized topic model
    """

    def __init__(self, input_size, bert_input_size, inference_type="zeroshot", num_topics=10, model_type='prodLDA',
                 hidden_sizes=(100, 100), activation='softplus', dropout=0.2, learn_priors=True, batch_size=64,
                 lr=2e-3, momentum=0.99, solver='adam', num_epochs=100, reduce_on_plateau=False, topic_prior_mean=0.0,
                 topic_prior_variance=None, num_data_loader_workers=0):
        """
        :param input_size: int, dimension of input
        :param bert_input_size: int, dimension of input that comes from BERT embeddings
        :param inference_type: string, you can choose between the contextual model and the combined model
        :param num_topics: int, number of topic components, (default 10)
        :param model_type: string, 'prodLDA' or 'LDA' (default 'prodLDA')
        :param hidden_sizes: tuple, length = n_layers, (default (100, 100))
        :param activation: string, 'softplus', 'relu', (default 'softplus')
        :param dropout: float, dropout to use (default 0.2)
        :param learn_priors: bool, make priors a learnable parameter (default True)
        :param batch_size: int, size of batch to use for training (default 64)
        :param lr: float, learning rate to use for training (default 2e-3)
        :param momentum: float, momentum to use for training (default 0.99)
        :param solver: string, optimizer 'adam' or 'sgd' (default 'adam')
        :param num_epochs: int, number of epochs to train for, (default 100)
        :param reduce_on_plateau: bool, reduce learning rate by 10x on plateau of 10 epochs (default False)
        :param num_data_loader_workers: int, number of data loader workers (default cpu_count). set it to 0 if you are using Windows
        """

        assert isinstance(input_size, int) and input_size > 0, \
            "input_size must by type int > 0."
        assert isinstance(num_topics, int) and input_size > 0, \
            "num_topics must by type int > 0."
        assert model_type in ['LDA', 'prodLDA'], \
            "model must be 'LDA' or 'prodLDA'."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu', 'sigmoid', 'swish', 'tanh', 'leakyrelu',
                              'rrelu', 'elu', 'selu'], \
            "activation must be 'softplus', 'relu', 'sigmoid', 'swish', 'leakyrelu'," \
            " 'rrelu', 'elu', 'selu' or 'tanh'."
        assert dropout >= 0, "dropout must be >= 0."
        # assert isinstance(learn_priors, bool), "learn_priors must be boolean."
        assert isinstance(batch_size, int) and batch_size > 0, \
            "batch_size must be int > 0."
        assert lr > 0, "lr must be > 0."
        assert isinstance(momentum, float) and momentum > 0 and momentum <= 1, \
            "momentum must be 0 < float <= 1."
        assert solver in ['adagrad', 'adam', 'sgd', 'adadelta', 'rmsprop'], \
            "solver must be 'adam', 'adadelta', 'sgd', 'rmsprop' or 'adagrad'"
        assert isinstance(reduce_on_plateau, bool), \
            "reduce_on_plateau must be type bool."
        assert isinstance(topic_prior_mean, float), \
            "topic_prior_mean must be type float"
        # and topic_prior_variance >= 0, \
        # assert isinstance(topic_prior_variance, float), \
        #    "topic prior_variance must be type float"

        self.input_size = input_size
        self.num_topics = num_topics
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.batch_size = batch_size
        self.lr = lr
        self.bert_size = bert_input_size
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.reduce_on_plateau = reduce_on_plateau
        self.num_data_loader_workers = num_data_loader_workers
        self.topic_prior_mean = topic_prior_mean
        self.topic_prior_variance = topic_prior_variance
        # init inference avitm network
        self.model = DecoderNetwork(
            input_size, self.bert_size, inference_type, num_topics, model_type, hidden_sizes, activation,
            dropout, self.learn_priors, self.topic_prior_mean, self.topic_prior_variance)
        self.early_stopping = EarlyStopping(patience=5, verbose=True)
        # init optimizer
        if self.solver == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(self.momentum, 0.99))
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=self.momentum)
        elif self.solver == 'adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=lr)
        elif self.solver == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        elif self.solver == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr, momentum=self.momentum)
        # init lr scheduler
        if self.reduce_on_plateau:
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=10)

        # performance attributes
        self.best_loss_train = float('inf')

        # training attributes
        self.model_dir = None
        self.train_data = None
        self.nn_epoch = None

        # learned topics
        self.best_components = None

        # Use cuda if available
        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False
        if self.USE_CUDA:
            self.model = self.model.cuda()

    def _loss(self, inputs, word_dists, prior_mean, prior_variance,
              posterior_mean, posterior_variance, posterior_log_variance):
        # KL term
        # var division term
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)
        # logvar det division term
        logvar_det_division = \
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        # combine terms
        KL = 0.5 * (var_division + diff_term - self.num_topics + logvar_det_division)
        # Reconstruction term
        RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)
        loss = KL + RL

        return loss.sum()

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0
        topic_doc_list = []
        for batch_samples in loader:
            # batch_size x vocab_size
            X = batch_samples['X']
            X = X.reshape(X.shape[0], -1)
            X_bert = batch_samples['X_bert']
            if self.USE_CUDA:
                X = X.cuda()
                X_bert = X_bert.cuda()

            # forward pass
            self.model.zero_grad()
            prior_mean, prior_variance, \
            posterior_mean, posterior_variance, posterior_log_variance, \
            word_dists, topic_word, topic_document = self.model(X, X_bert)
            topic_doc_list.extend(topic_document)

            # backward pass
            loss = self._loss(X, word_dists, prior_mean, prior_variance,
                              posterior_mean, posterior_variance, posterior_log_variance)
            loss.backward()
            self.optimizer.step()

            # compute train loss
            samples_processed += X.size()[0]
            train_loss += loss.item()

        train_loss /= samples_processed

        return samples_processed, train_loss, topic_word, topic_doc_list

    def _validation(self, loader):
        """Train epoch."""
        self.model.eval()
        val_loss = 0
        samples_processed = 0
        for batch_samples in loader:
            # batch_size x vocab_size
            X = batch_samples['X']
            X = X.reshape(X.shape[0], -1)
            X_bert = batch_samples['X_bert']

            if self.USE_CUDA:
                X = X.cuda()
                X_bert = X_bert.cuda()

            # forward pass
            self.model.zero_grad()
            prior_mean, prior_variance, \
            posterior_mean, posterior_variance, posterior_log_variance, \
            word_dists, topic_word, topic_document = self.model(X, X_bert)

            loss = self._loss(X, word_dists, prior_mean, prior_variance,
                              posterior_mean, posterior_variance, posterior_log_variance)

            # compute train loss
            samples_processed += X.size()[0]
            val_loss += loss.item()

        val_loss /= samples_processed

        return samples_processed, val_loss

    def fit(self, train_dataset, validation_dataset, save_dir=None, verbose=True):
        """
        Train the CTM model.

        :param train_dataset: PyTorch Dataset class for training data.
        :param validation_dataset: PyTorch Dataset class for validation data
        :param save_dir: directory to save checkpoint models to.
        :param verbose: verbose
        """
        # Print settings to output file
        if verbose:
            print("Settings: \n\
                   N Components: {}\n\
                   Topic Prior Mean: {}\n\
                   Topic Prior Variance: {}\n\
                   Model Type: {}\n\
                   Hidden Sizes: {}\n\
                   Activation: {}\n\
                   Dropout: {}\n\
                   Learn Priors: {}\n\
                   Learning Rate: {}\n\
                   Momentum: {}\n\
                   Reduce On Plateau: {}\n\
                   Save Dir: {}".format(
                self.num_topics, self.topic_prior_mean,
                self.topic_prior_variance, self.model_type,
                self.hidden_sizes, self.activation, self.dropout, self.learn_priors,
                self.lr, self.momentum, self.reduce_on_plateau, save_dir))

        self.model_dir = save_dir
        self.train_data = train_dataset
        self.validation_data = validation_dataset

        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_data_loader_workers)

        # init training variables
        train_loss = 0
        samples_processed = 0

        # train loop
        for epoch in range(self.num_epochs):
            self.nn_epoch = epoch
            # train epoch
            s = datetime.datetime.now()
            sp, train_loss, topic_word, topic_document = self._train_epoch(train_loader)
            samples_processed += sp
            e = datetime.datetime.now()

            if verbose:
                print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                    epoch + 1, self.num_epochs, samples_processed,
                    len(self.train_data) * self.num_epochs, train_loss, e - s))

            self.best_components = self.model.beta
            self.final_topic_word = topic_word
            self.final_topic_document = topic_document
            self.best_loss_train = train_loss

            validation_loader = DataLoader(
                self.validation_data, batch_size=self.batch_size, shuffle=True,
                num_workers=self.num_data_loader_workers)
            # train epoch
            s = datetime.datetime.now()
            val_samples_processed, val_loss = self._validation(validation_loader)
            e = datetime.datetime.now()

            # report
            print("Epoch: [{}/{}]\tSamples: [{}/{}]\tValidation Loss: {}\tTime: {}".format(
                epoch + 1, self.num_epochs, val_samples_processed,
                len(self.validation_data) * self.num_epochs, val_loss, e - s))

            if np.isnan(val_loss) or np.isnan(train_loss):
                break
            else:
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    if save_dir is not None:
                        self.save(save_dir)
                    break

    def predict(self, dataset):
        """Predict input."""
        self.model.eval()

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_data_loader_workers)

        topic_document_mat = []
        with torch.no_grad():
            for batch_samples in loader:
                # batch_size x vocab_size
                X = batch_samples['X']
                X = X.reshape(X.shape[0], -1)
                X_bert = batch_samples['X_bert']

                if self.USE_CUDA:
                    X = X.cuda()
                    X_bert = X_bert.cuda()
                # forward pass
                self.model.zero_grad()
                _, _, _, _, _, _, _, topic_document = self.model(X, X_bert)
                topic_document_mat.append(topic_document)

        results = self.get_info()
        results['test-topic-document-matrix'] = np.asarray(self.get_thetas(dataset)).T

        return results

    def get_topic_word_mat(self):
        top_wor = self.final_topic_word.cpu().detach().numpy()
        return top_wor

    def get_topic_document_mat(self):
        top_doc = self.final_topic_document
        top_doc_arr = np.array([i.cpu().detach().numpy() for i in top_doc])
        return top_doc_arr

    def get_topics(self, k=10):
        """
        Retrieve topic words.

        Args
            k : (int) number of words to return per topic, default 10.
        """
        assert k <= self.input_size, "k must be <= input size."
        component_dists = self.best_components
        topics = defaultdict(list)
        topics_list = []
        if self.num_topics is not None:
            for i in range(self.num_topics):
                _, idxs = torch.topk(component_dists[i], k)
                component_words = [self.train_data.idx2token[idx]
                                   for idx in idxs.cpu().numpy()]
                topics[i] = component_words
                topics_list.append(component_words)

        return topics_list

    def get_info(self):
        info = {}
        topic_word = self.get_topics()
        topic_word_dist = self.get_topic_word_mat()
        topic_document_dist = self.get_topic_document_mat()
        info['topics'] = topic_word

        info['topic-document-matrix'] = np.asarray(self.get_thetas(self.train_data)).T

        info['topic-word-matrix'] = topic_word_dist
        return info

    def _format_file(self):
        model_dir = "AVITM_nc_{}_tpm_{}_tpv_{}_hs_{}_ac_{}_do_{}_lr_{}_mo_{}_rp_{}". \
            format(self.num_topics, 0.0, 1 - (1. / self.num_topics),
                   self.model_type, self.hidden_sizes, self.activation,
                   self.dropout, self.lr, self.momentum,
                   self.reduce_on_plateau)
        return model_dir

    def save(self, models_dir=None):
        """
        Save model.

        :param models_dir: path to directory for saving NN models.
        """
        if (self.model is not None) and (models_dir is not None):

            model_dir = self._format_file()
            if not os.path.isdir(os.path.join(models_dir, model_dir)):
                os.makedirs(os.path.join(models_dir, model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(models_dir, model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.model.state_dict(),
                            'dcue_dict': self.__dict__}, file)

    def load(self, model_dir, epoch):
        """
        Load a previously trained model.

        :param model_dir: directory where models are saved.
        :param epoch: epoch of model to load.
        """
        epoch_file = "epoch_" + str(epoch) + ".pth"
        model_file = os.path.join(model_dir, epoch_file)
        with open(model_file, 'rb') as model_dict:
            checkpoint = torch.load(model_dict)

        for (k, v) in checkpoint['dcue_dict'].items():
            setattr(self, k, v)

        self.model.load_state_dict(checkpoint['state_dict'])

    def get_thetas(self, dataset):
        """Predict input."""
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_data_loader_workers)
        with torch.no_grad():
            collect_theta = []
            for batch_samples in loader:
                # batch_size x vocab_size
                X = batch_samples['X']
                X = X.reshape(X.shape[0], -1)
                X_bert = batch_samples['X_bert']
                if self.USE_CUDA:
                    X = X.cuda()
                    X_bert = X_bert.cuda()

                # forward pass
                self.model.zero_grad()
                collect_theta.extend(self.model.get_theta(X, X_bert).cpu().numpy().tolist())
            return collect_theta
