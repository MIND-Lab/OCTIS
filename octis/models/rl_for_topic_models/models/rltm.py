import datetime
import os
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from octis.models.rl_for_topic_models.networks.decoder_network import (
    DecoderNetwork)


class RLTM(object):
    """Class to train the reinforcement learning topic model
    """

    def __init__(
        self, input_size, bert_size, num_topics=10, hidden_sizes=(128, 128),
        activation='gelu', inference_dropout=0.2, policy_dropout=0.0,
        batch_size=256, lr=3e-4, momentum=0.9, solver='adamw', num_epochs=200,
        num_samples=10, reduce_on_plateau=False, top_words=10,
        num_data_loader_workers=0, weight_decay=0.01, kl_multiplier=1.0,
        grad_norm_clip=1.0):

        """
        :param input_size: int, dimension of input
        :param bert_input_size: int, dimension of BERT input
        :param num_topics: int, number of topic components, (default 10)
        :param hidden_sizes: tuple, length = n_layers, (default (128, 128))
        :param activation: string, 'softplus', 'relu', 'sigmoid', 'swish',
            'tanh', 'leakyrelu', 'rrelu', 'elu', 'selu', 'gelu' (default 'gelu')
        :param inference_dropout: float, inference dropout to use (default 0.2)
        :param policy_dropout: float, policy dropout to use (default 0.0)
        :param batch_size: int, size of batch to use for training (default 256)
        :param lr: float, learning rate to use for training (default 3e-4)
        :param momentum: float, momentum to use for training (default 0.9)
        :param solver: string, optimizer 'adagrad', 'adam', 'sgd', 'adadelta',
            'rmsprop', 'adamw' (default 'adamw')
        :param num_samples: int, number of times theta needs to be sampled
        :param num_epochs: int, number of epochs to train for, (default 200)
        :param reduce_on_plateau: bool, reduce learning rate by 10x on plateau
            of 10 epochs (default False)
        :param num_data_loader_workers: int, number of data loader workers
            (default cpu_count). set it to 0 if you are using Windows
        :param weight_decay: float, L2 regularization on model weights (default 0.01)
        :param kl_multiplier: float or int, multiplier on the KL
            divergence (default 1.0)
        :param grad_norm_clip: float or None; clip gradient norms (default 1.0)
        """

        assert isinstance(input_size, int) and input_size > 0, \
            "input_size must by type int > 0."
        assert isinstance(bert_size, int) and bert_size > 0, \
            "bert_size must by type int > 0."
        assert (isinstance(num_topics, int) or isinstance(
            num_topics, np.int64)) and num_topics > 0, \
            "num_topics must by type int > 0."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu', 'sigmoid', 'tanh', 'leakyrelu',
                              'rrelu', 'elu', 'selu', 'gelu'], \
            "activation must be 'softplus', 'relu', 'sigmoid', 'tanh'," \
            " 'leakyrelu', 'rrelu', 'elu', 'selu', or 'gelu'."
        assert inference_dropout >= 0, "inference dropout must be >= 0."
        assert policy_dropout >= 0, "policy dropout must be >= 0."
        assert isinstance(batch_size, int) and batch_size > 0, \
            "batch_size must be int > 0."
        assert lr > 0, "lr must be > 0."
        assert isinstance(
            momentum, float) and momentum > 0 and momentum <= 1, \
            "momentum must be 0 < float <= 1."
        assert solver in ['adagrad', 'adam', 'sgd', 'adadelta', 'rmsprop',
                          'adamw'], "solver must be 'adam', 'adadelta', \
                        'sgd', 'rmsprop', 'adagrad', or 'adamw'"
        assert isinstance(num_epochs, int) and num_epochs > 0, \
            "num_epochs must be int > 0"
        assert isinstance(num_samples, int) and num_samples > 0, \
            "num_samples must be int > 0"
        assert isinstance(reduce_on_plateau, bool), \
            "reduce_on_plateau must be type bool."
        assert isinstance(top_words, int) and top_words > 0, \
            "top_words must be int > 0"
        assert isinstance(num_data_loader_workers, int) \
            and num_data_loader_workers >= 0, \
            "num_data_loader_workers must be int >= 0"
        assert weight_decay >= 0, "weight_decay must be >= 0"
        assert isinstance(kl_multiplier, float) or isinstance(kl_multiplier, int), \
            "kl_multiplier must be a float or int"
        assert isinstance(grad_norm_clip, float) or grad_norm_clip is None, \
            "grad_norm_clip must be a float or None"
        if grad_norm_clip is not None:
            assert grad_norm_clip > 0, "grad_norm_clip must be > 0"

        self.input_size = input_size
        self.num_topics = num_topics
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.inference_dropout = inference_dropout
        self.policy_dropout = policy_dropout
        self.batch_size = batch_size
        self.lr = lr
        self.num_samples = num_samples
        self.top_words = top_words
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.reduce_on_plateau = reduce_on_plateau
        self.num_data_loader_workers = num_data_loader_workers
        self.grad_norm_clip = grad_norm_clip

        # init decoder network
        model = DecoderNetwork(
            input_size, bert_size, num_topics, hidden_sizes, activation,
            inference_dropout, policy_dropout, kl_multiplier)

        # init optimizer
        if self.solver == 'adamw':
            self.optimizer = self._configure_adamw(model, weight_decay, lr,
                betas=(self.momentum, 0.999))
        if self.solver == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(
                self.momentum, 0.999))
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=self.momentum)
        elif self.solver == 'adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=lr)
        elif self.solver == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        elif self.solver == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(), lr=lr, momentum=self.momentum)
        
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
        self.model = model
        if torch.cuda.is_available():
            self.USE_CUDA = True
            self.model = self.model.cuda()
        else:
            self.USE_CUDA = False            
    
    @staticmethod
    def _configure_adamw(model, weight_decay, lr, betas):
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        whitelist_weight_names = ()
        blacklist_weight_names = ('prior_mean', 'prior_variance', 'beta')

        decay = set()
        no_decay = set()
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                # don't decay biases
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                # decay weights according to white/blacklist
                elif pn.endswith('weight'):
                    if isinstance(m, whitelist_weight_modules):
                        decay.add(fpn)
                    elif isinstance(m, blacklist_weight_modules):
                        no_decay.add(fpn)
                else:
                    if fpn in whitelist_weight_names:
                        decay.add(fpn)
                    elif fpn in blacklist_weight_names:
                        no_decay.add(fpn)
        param_dict = {pn: p for pn, p in model.named_parameters()}

        # for decay and no decay sets, ensure no intersection and union contains all parameters
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f'parameters {inter_params} made it into both decay and no_decay set'
        assert len(param_dict.keys() - union_params) == 0, \
            f'parameters {param_dict.keys() - union_params} were not separated into either decay or no_decay set'

        optim_groups = [
            {'params': [param_dict[pn] for pn in sorted(list(decay))], 'weight_decay': weight_decay},
            {'params': [param_dict[pn] for pn in sorted(list(no_decay))], 'weight_decay': 0.0}
        ]

        return optim.AdamW(optim_groups, lr=lr, betas=betas)

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
            loss = self.model(X, X_bert)
            topic_doc_list.extend(F.softmax(self.model.action, dim=-1))

            # backward pass
            loss.backward()
            if self.grad_norm_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            self.optimizer.step()

            # compute train loss
            samples_processed += X.size()[0]
            train_loss += loss.item()

        train_loss /= samples_processed

        return samples_processed, train_loss, self.model.beta, topic_doc_list

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
            loss = self.model(X, X_bert)

            # compute train loss
            samples_processed += X.size()[0]
            val_loss += loss.item()

        val_loss /= samples_processed

        return samples_processed, val_loss

    def fit(self, train_dataset, validation_dataset=None,
            save_dir=None, verbose=True):
        """
        Train the RLTM model.

        :param train_dataset: PyTorch Dataset class for training data.
        :param validation_dataset: PyTorch Dataset class for validation data
        :param save_dir: directory to save checkpoint models to.
        :param verbose: verbose
        """
        # Print settings to output file
        if verbose:
            print("Settings: \n\
                   N Components: {}\n\
                   Hidden Sizes: {}\n\
                   Activation: {}\n\
                   Inference Dropout: {}\n\
                   Policy Dropout: {}\n\
                   Batch Size: {}\n\
                   Learning Rate: {}\n\
                   Momentum: {}\n\
                   Reduce On Plateau: {}\n\
                   Save Dir: {}".format(
                self.num_topics, self.hidden_sizes, self.activation,
                self.inference_dropout, self.policy_dropout, self.batch_size,
                self.lr, self.momentum, self.reduce_on_plateau, save_dir))

        self.model_dir = save_dir
        self.train_data = train_dataset
        self.validation_data = validation_dataset

        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_data_loader_workers)

        # init training variables
        train_loss = 0
        samples_processed = 0

        # train loop
        for epoch in range(self.num_epochs):
            self.nn_epoch = epoch
            # train epoch
            s = datetime.datetime.now()
            sp, train_loss, topic_word, topic_document = self._train_epoch(
                train_loader)
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
            if self.validation_data is not None:
                validation_loader = DataLoader(
                    self.validation_data, batch_size=self.batch_size,
                    shuffle=True, num_workers=self.num_data_loader_workers)
                # train epoch
                s = datetime.datetime.now()
                val_samples_processed, val_loss = self._validation(
                    validation_loader)
                e = datetime.datetime.now()

                if verbose:
                    print(
                        "Epoch: [{}/{}]\tSamples: [{}/{}]"
                        "\tValidation Loss: {}\tTime: {}".format(
                            epoch + 1, self.num_epochs, val_samples_processed,
                            len(self.validation_data) * self.num_epochs,
                            val_loss, e - s))

                if np.isnan(val_loss) or np.isnan(train_loss):
                    print("loss is NaN")
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
                _ = self.model(X, X_bert)
                topic_document_mat.append(F.softmax(self.model.action, dim=-1))

        results = self.get_info()
        results['test-topic-document-matrix'] = np.asarray(
            self.get_thetas(dataset)).T

        return results

    def get_topic_word_mat(self):
        top_wor = self.final_topic_word.cpu().detach().numpy()
        return top_wor

    def get_topic_document_mat(self):
        top_doc = self.final_topic_document
        top_doc_arr = np.array([i.cpu().detach().numpy() for i in top_doc])
        return top_doc_arr

    def get_topics(self):
        """
        Retrieve topic words.

        """
        assert self.top_words <= self.input_size, "top_words must be <= input size."  # noqa
        component_dists = self.best_components
        topics = defaultdict(list)
        topics_list = []
        if self.num_topics is not None:
            for i in range(self.num_topics):
                _, idxs = torch.topk(component_dists[i], self.top_words)
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

        info['topic-document-matrix'] = np.asarray(
            self.get_thetas(self.train_data)).T

        info['topic-word-matrix'] = topic_word_dist
        return info

    def _format_file(self):
        model_dir = (
            "RLTM_nc_{}_hs_{}_ac_{}_id_{}_"
            "pd_{}_bs_{}_lr_{}_mo_{}_rp_{}".format(
                self.num_topics, self.hidden_sizes, self.activation,
                self.inference_dropout, self.policy_dropout, self.batch_size,
                self.lr, self.momentum, self.reduce_on_plateau))
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
        """
        Get the document-topic distribution for a dataset of topics. 
        Includes multiple sampling to reduce variation via
        the parameter num_samples.
        :param dataset: a PyTorch Dataset containing the documents
        """
        self.model.eval()

        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_data_loader_workers)
        final_thetas = []
        for sample_index in range(self.num_samples):
            with torch.no_grad():
                collect_theta = []
                for batch_samples in loader:
                    # batch_size x vocab_size
                    x_bert = batch_samples['X_bert']
                    if self.USE_CUDA:
                        x_bert = x_bert.cuda()
                    # forward pass
                    self.model.zero_grad()
                    collect_theta.extend(
                        self.model.get_topic_distribution(x_bert).cpu().numpy().tolist())

                final_thetas.append(np.array(collect_theta))
        return np.sum(final_thetas, axis=0) / self.num_samples
