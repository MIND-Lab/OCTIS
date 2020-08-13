
from __future__ import print_function

import argparse
import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
import sys
import matplotlib.pyplot as plt 
from models.ETM import data
import scipy.io
from sklearn.feature_extraction.text import CountVectorizer
from torch import nn, optim
from torch.nn import functional as F
from models.ETM import etm
from models.model import Abstract_Model
from models.ETM.utils import nearest_neighbors, get_topic_coherence, get_topic_diversity

class ETM_Wrapper(Abstract_Model):

    def __init__(self):
        self.hyperparameters = {}

    def train_model(self, dataset, hyperparameters, top_words=10, topic_word_matrix=True,
                    topic_document_matrix=True, embeddings=None, train_embeddings=True):
        print(train_embeddings, "pre_set_model")
        self.set_model(dataset, hyperparameters, embeddings, train_embeddings)
        print(train_embeddings, "post_set_model")
        self.bool_topic_doc = topic_document_matrix
        self.bool_topic_word = topic_word_matrix
        self.top_word = top_words
        best_epoch = 0
        best_val_ppl = 1e9
        all_val_ppls = []
        #print('\n')
        #print('Visualizing model quality before training...')
        #visualize(self.model)
        #print('\n')
        for epoch in range(0, self.hyperparameters['num_epochs']):
            self._train_epoch(epoch)
        if self.use_partitions:
            result = self.inference()
        else:
            result = self.get_info()
    
        return result

    def set_model(self, dataset, hyperparameters, embeddings, train_embeddings):
        if self.use_partitions:
            data = dataset.get_partitioned_corpus()
            X_train = data[0]
            X_test = data[1]
            data_corpus_train = [','.join(i) for i in X_train]
            data_corpus_test = [','.join(i) for i in X_test]
            self.train_tokens, self.train_counts, self.test_tokens, self.test_counts, \
            self.vocab_size, self.vocab = self.preprocess(data_corpus_train, data_corpus_test)
        else:
            data_corpus = [','.join(i) for i in dataset.get_corpus()]
            self.train_tokens, self.train_counts, self.vocab_size, self.vocab = self.preprocess(data_corpus, None)

        self.num_docs_train = self.train_tokens.shape[1]
        self.embeddings = embeddings
        self.train_embeddings = train_embeddings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(0)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)

        self.set_default_hyperparameters(hyperparameters)
        ## define model and optimizer
        print(self.train_embeddings, "pre_set_default_hyp")

        self.model = etm.ETM(num_topics=self.hyperparameters['num_topics'],
                             vocab_size=self.vocab_size,
                             t_hidden_size=self.hyperparameters['t_hidden_size'],
                             rho_size=self.hyperparameters['rho_size'],
                             emsize=self.hyperparameters['emb_size'],
                             theta_act=self.hyperparameters['theta_act'],
                             embeddings=self.embeddings,
                             train_embeddings=self.train_embeddings,
                             enc_drop=self.hyperparameters['enc_drop']).to(self.device)
        print('model: {}'.format(self.model))

        if self.hyperparameters['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparameters['lr'])
        elif self.hyperparameters['optimizer'] == 'adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.hyperparameters['lr'])
        elif self.hyperparameters['optimizer'] == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.hyperparameters['lr'])
        elif self.hyperparameters['optimizer'] == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.hyperparameters['lr'])
        elif self.hyperparameters['optimizer'] == 'asgd':
            self.optimizer = optim.ASGD(self.model.parameters(), lr=self.hyperparameters['lr'], t0=0, lambd=0.)
        else:
            print('Defaulting to vanilla SGD')
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.hyperparameters['lr'])

    def _train_epoch(self, epoch):
        self.data_list = []
        self.model.train()
        acc_loss = 0
        acc_kl_theta_loss = 0
        cnt = 0
        indices = torch.arange(0, self.num_docs_train)
        indices = torch.split(indices, self.hyperparameters['batch_size'])
        for idx, ind in enumerate(indices):
            self.optimizer.zero_grad()
            self.model.zero_grad()
            data_batch = data.get_batch(self.train_tokens, self.train_counts, ind, self.vocab_size,
                                        self.hyperparameters['emb_size'], self.device)
            sums = data_batch.sum(1).unsqueeze(1)
            bow_norm = True
            if bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            recon_loss, kld_theta = self.model(data_batch, normalized_data_batch)
            total_loss = recon_loss + kld_theta
            total_loss.backward()
            # if args.clip > 0:
            #    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
            self.optimizer.step()

            acc_loss += torch.sum(recon_loss).item()
            acc_kl_theta_loss += torch.sum(kld_theta).item()
            cnt += 1
            log_interval = 2
            if idx % log_interval == 0 and idx > 0:
                cur_loss = round(acc_loss / cnt, 2)
                cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
                cur_real_loss = round(cur_loss + cur_kl_theta, 2)

            # print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
            #     epoch, idx, len(indices), self.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
            self.data_list.append(normalized_data_batch)
        cur_loss = round(acc_loss / cnt, 2)
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
        cur_real_loss = round(cur_loss + cur_kl_theta, 2)
        print('*' * 100)
        print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
            epoch + 1, self.optimizer.
                param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
        print('*' * 100)

    def get_info(self):
        topic_w = []
        self.model.eval()
        info = {}
        with torch.no_grad():
            theta, _ = self.model.get_theta(torch.cat(self.data_list))
            gammas = self.model.get_beta()
            for k in range(self.hyperparameters['num_topics']):
                gamma = gammas[k]
                top_words = list(gamma.cpu().numpy().argsort()[-self.top_word:][::-1])
                topic_words = [self.vocab[a] for a in top_words]
                #print('Topic {}: {}'.format(k, topic_words))
                topic_w.append(topic_words)
        if (self.bool_topic_doc == True) and (self.bool_topic_word == True):
            info['topics'] = topic_w
            info['topic-word-matrix'] = self.model.get_beta().cpu().detach().numpy()
            info['topic-document-matrix'] = theta.cpu().detach().numpy()
        elif (self.bool_topic_doc == True) and (self.bool_topic_word == False):
            info['topics'] = topic_w
            info['topic-document-matrix'] = theta.cpu().detach().numpy()
        elif (self.bool_topic_doc == False) and (self.bool_topic_word == True):
            info['topics'] = topic_w
            info['topic-word-matrix'] = self.model.get_beta().cpu().detach().numpy()
        else:
            info['topics'] = topic_w
        return info

    def inference(self):
        
        assert isinstance(self.use_partitions, bool) and self.use_partitions == True
        
        topic_d = []
        self.model.eval()
        indices = torch.arange(0, self.test_tokens.shape[1])
        indices = torch.split(indices, self.hyperparameters['batch_size'])

        for idx, ind in enumerate(indices):
            data_batch = data.get_batch(self.test_tokens, self.test_counts, ind, self.vocab_size,
                                        self.hyperparameters['emb_size'], self.device)
            sums = data_batch.sum(1).unsqueeze(1)
            bow_norm = True
            if bow_norm:
                normalized_data_batch = data_batch / sums
            theta, _ = self.model.get_theta(normalized_data_batch)
            topic_d.append(theta.cpu().detach().numpy())

        info = self.get_info()
        emp_array = np.empty((0, self.hyperparameters['num_topics']))
        topic_doc = np.asarray(topic_d)
        length = topic_doc.shape[0]
        if self.bool_topic_doc:
            # batch concatenation
            for i in range(length):
                emp_array = np.concatenate([emp_array, topic_doc[i]])
            info['test-topic-document-matrix'] = emp_array

        return info


    def get_test(self):
        tok = self.test_tokens
        cou = self.test_counts
        return tok,cou

    def set_default_hyperparameters(self, hyperparameters):

        self.hyperparameters['num_topics'] = hyperparameters.get('num_topics', self.hyperparameters.get('num_topics', 10))
        self.hyperparameters['num_epochs'] = hyperparameters.get('num_epochs', self.hyperparameters.get('num_epochs', 20))
        self.hyperparameters['t_hidden_size'] = hyperparameters.get('t_hidden_size', self.hyperparameters.get('t_hidden_size',800))
        self.hyperparameters['rho_size'] = hyperparameters.get('rho_size',self.hyperparameters.get('rho_size', 300))
        self.hyperparameters['emb_size'] = hyperparameters.get('emb_size', self.hyperparameters.get('emb_size', 300))
        self.hyperparameters['theta_act'] = hyperparameters.get('theta_act', self.hyperparameters.get('theta_act','relu'))
        self.hyperparameters['enc_drop'] = hyperparameters.get('enc_drop',self.hyperparameters.get('enc_drop',0.0))
        self.hyperparameters['lr'] = hyperparameters.get('lr', self.hyperparameters.get('lr',0.005))
        self.hyperparameters['optimizer'] = hyperparameters.get('optimizer', self.hyperparameters.get('optimizer', 'adam'))
        self.hyperparameters['batch_size'] = hyperparameters.get('batch_size', self.hyperparameters.get('batch_size',128))

    def partitioning(self, use_partitions=False):
        if use_partitions:
            self.use_partitions = True
        else:
            self.use_partitions = False

    @staticmethod
    def preprocess(dataset, test=None):

        def split_bow(bow_in, n_docs):
            indices = np.asarray([np.asarray([w for w in bow_in[doc,:].indices]) for doc in range(n_docs)])
            counts = np.asarray([np.asarray([c for c in bow_in[doc,:].data]) for doc in range(n_docs)])
            return np.expand_dims(indices, axis = 0), np.expand_dims(counts, axis = 0)

        if test is not None:
            vec = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
            X_train = vec.fit_transform(dataset)
            X_test = vec.transform(test)
            idx2token = {v: k for (k, v) in vec.vocabulary_.items()}
            vocab_size = len(idx2token.keys())
            vocab = vec.vocabulary_.keys()

            X_train_tokens, X_train_count = split_bow(X_train, X_train.shape[0])
            X_test_tokens, X_test_count = split_bow(X_test, X_test.shape[0])

            return X_train_tokens, X_train_count,X_test_tokens, X_test_count, vocab_size, list(vocab)

        else:
            vec = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
            X_train = vec.fit_transform(dataset)
            idx2token = {v: k for (k, v) in vec.vocabulary_.items()}
            vocab_size = len(idx2token.keys())
            vocab = vec.vocabulary_.keys()
            X_train_tokens, X_train_count = split_bow(X_train, X_train.shape[0])

            return X_train_tokens, X_train_count, vocab_size, list(vocab)
