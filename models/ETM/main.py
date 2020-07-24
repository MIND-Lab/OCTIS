#/usr/bin/python

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

    def train_model(self, dataset, hyperparameters,  embeddings = None, train_embeddings = True, top_words = 10):
        self.set_model(dataset, hyperparameters, embeddings, train_embeddings)
        self.top_word = top_words
        best_epoch = 0
        best_val_ppl = 1e9
        all_val_ppls = []
        #print('\n')
        #print('Visualizing model quality before training...')
        #visualize(self.model)
        #print('\n')
        for epoch in range(0, self.hyperparameters['epochs']):
            self.train_epoch(epoch)
        return self.get_info()

    def set_model(self, dataset, hyperparameters, embeddings, train_embeddings):
        self.train_tokens, self.train_counts, self.vocab_size, self.vocab = self.preprocess(dataset)
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
        self.model = etm.ETM(self.hyperparameters['num_topics'], self.vocab_size,
                             self.hyperparameters['t_hidden_size'],
                             self.hyperparameters['rho_size'], self.hyperparameters['emb_size'],
                             self.hyperparameters['theta_act'], self.embeddings, self.train_embeddings,
                             self.hyperparameters['enc_drop']).to(self.device)
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

    def train_epoch(self, epoch):
        self.data_list = []
        self.model.train()
        acc_loss = 0
        acc_kl_theta_loss = 0
        cnt = 0
        indices = torch.randperm(self.num_docs_train)
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
            epoch, self.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
        print('*' * 100)

    def get_info(self):
        topic_w = []
        self.model.eval()
        info = {}
        theta, _ = self.model.get_theta(torch.cat(self.data_list))
        with torch.no_grad():
            topics_words = []
            gammas = self.model.get_beta()
            for k in range(self.hyperparameters['num_topics']):
                gamma = gammas[k]
                top_words = list(gamma.cpu().numpy().argsort()[-self.top_word:][::-1])
                topic_words = [self.vocab[a] for a in top_words]
                topics_words.append(' '.join(topic_words))
                #print('Topic {}: {}'.format(k, topic_words))
                topic_w.append(topic_words)
        info['topics'] = topic_w
        info['topic-word-matrix'] = self.model.get_beta().cpu().detach().numpy()
        info['topic-document-matrix'] = theta.cpu().detach().numpy()
        return info
    #beta topic-word distribution
    #theta topic-document distribution
    def set_default_hyperparameters(self, hyperparameters):
        self.hyperparameters['num_topics'] = hyperparameters.get('num_topics', 10)
        self.hyperparameters['epochs'] = hyperparameters.get('epochs', 20)
        self.hyperparameters['t_hidden_size'] = hyperparameters.get('t_hidden_size', 800)
        self.hyperparameters['rho_size'] = hyperparameters.get('rho_size', 300)
        self.hyperparameters['emb_size'] = hyperparameters.get('emb_size', 300)
        self.hyperparameters['theta_act'] = hyperparameters.get('theta_act', 'relu')
        self.hyperparameters['enc_drop'] = hyperparameters.get('enc_drop', 0.0)
        self.hyperparameters['lr'] = hyperparameters.get('lr', 0.005)
        self.hyperparameters['optimizer'] = hyperparameters.get('optimizer', 'adam')
        self.hyperparameters['batch_size'] = hyperparameters.get('batch_size', 128)

    @staticmethod
    def preprocess(dataset):
        data_corpus = [','.join(i) for i in dataset.get_corpus()]
        vec = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
        X = vec.fit_transform(data_corpus)
        idx2token = {v: k for (k, v) in vec.vocabulary_.items()}
        vocab_size = len(idx2token.keys())
        vocab = vec.vocabulary_.keys()
        def split_bow(bow_in, n_docs):
            indices = np.asarray([np.asarray([w for w in bow_in[doc,:].indices]) for doc in range(n_docs)])
            counts = np.asarray([np.asarray([c for c in bow_in[doc,:].data]) for doc in range(n_docs)])
            return np.expand_dims(indices, axis = 0), np.expand_dims(counts, axis = 0)

        bow_ts_h1_tokens, bow_ts_h1_count = split_bow(X, X.shape[0])
        return bow_ts_h1_tokens, bow_ts_h1_count, vocab_size, list(vocab)
"""

def visualize(m, show_emb=True):
    if not os.path.exists('./results'):
        os.makedirs('./results')

    m.eval()

    queries = ['andrew', 'computer', 'sports', 'religion', 'man', 'love', 
                'intelligence', 'money', 'politics', 'health', 'people', 'family']

    ## visualize topics using monte carlo
    with torch.no_grad():
        print('#'*100)
        print('Visualize topics...')
        topics_words = []
        gammas = m.get_beta()
        for k in range(args.num_topics):
            gamma = gammas[k]
            top_words = list(gamma.cpu().numpy().argsort()[-args.num_words+1:][::-1])
            topic_words = [vocab[a] for a in top_words]
            topics_words.append(' '.join(topic_words))
            print('Topic {}: {}'.format(k, topic_words))

        if show_emb:
            ## visualize word embeddings by using V to get nearest neighbors
            print('#'*100)
            print('Visualize word embeddings by using output embedding matrix')
            try:
                embeddings = m.rho.weight  # Vocab_size x E
            except:
                embeddings = m.rho         # Vocab_size x E
            neighbors = []
            for word in queries:
                print('word: {} .. neighbors: {}'.format(
                    word, nearest_neighbors(word, embeddings, vocab)))
            print('#'*100)
"""
"""
def evaluate(m, source, tc=False, td=False):
    m.eval()
    with torch.no_grad():
        if source == 'val':
            indices = torch.split(torch.tensor(range(args.num_docs_valid)), args.eval_batch_size)
            tokens = valid_tokens
            counts = valid_counts
        else: 
            indices = torch.split(torch.tensor(range(args.num_docs_test)), args.eval_batch_size)
            tokens = test_tokens
            counts = test_counts

        ## get beta here
        beta = m.get_beta()

        ### do dc and tc here
        acc_loss = 0
        cnt = 0
        indices_1 = torch.split(torch.tensor(range(args.num_docs_test_1)), args.eval_batch_size)
        for idx, ind in enumerate(indices_1):
            ## get theta from first half of docs
            data_batch_1 = data.get_batch(test_1_tokens, test_1_counts, ind, args.vocab_size, device)
            sums_1 = data_batch_1.sum(1).unsqueeze(1)
            if args.bow_norm:
                normalized_data_batch_1 = data_batch_1 / sums_1
            else:
                normalized_data_batch_1 = data_batch_1
            theta, _ = m.get_theta(normalized_data_batch_1)

            ## get prediction loss using second half
            data_batch_2 = data.get_batch(test_2_tokens, test_2_counts, ind, args.vocab_size, device)
            sums_2 = data_batch_2.sum(1).unsqueeze(1)
            res = torch.mm(theta, beta)
            preds = torch.log(res)
            recon_loss = -(preds * data_batch_2).sum(1)
            
            loss = recon_loss / sums_2.squeeze()
            loss = loss.mean().item()
            acc_loss += loss
            cnt += 1
        cur_loss = acc_loss / cnt
        ppl_dc = round(math.exp(cur_loss), 1)
        print('*'*100)
        print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
        print('*'*100)
        if tc or td:
            beta = beta.data.cpu().numpy()
            if tc:
                print('Computing topic coherence...')
                get_topic_coherence(beta, train_tokens, vocab)
            if td:
                print('Computing topic diversity...')
                get_topic_diversity(beta, 25)
        return ppl_dc
"""


"""
        val_ppl = evaluate(model, 'val')
        if val_ppl < best_val_ppl:
            with open(ckpt, 'wb') as f:
                torch.save(model, f)
            best_epoch = epoch
            best_val_ppl = val_ppl
        else:
            ## check whether to anneal lr
            lr = optimizer.param_groups[0]['lr']
            if args.anneal_lr and (len(all_val_ppls) > args.nonmono and val_ppl > min(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
                optimizer.param_groups[0]['lr'] /= args.lr_factor
        if epoch % args.visualize_every == 0:
            visualize(model)
        all_val_ppls.append(val_ppl)
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    val_ppl = evaluate(model, 'val')
else:   
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        ## get document completion perplexities
        test_ppl = evaluate(model, 'test', tc=args.tc, td=args.td)

        ## get most used topics
        indices = torch.tensor(range(args.num_docs_train))
        indices = torch.split(indices, args.batch_size)
        thetaAvg = torch.zeros(1, args.num_topics).to(device)
        thetaWeightedAvg = torch.zeros(1, args.num_topics).to(device)
        cnt = 0
        for idx, ind in enumerate(indices):
            data_batch = data.get_batch(train_tokens, train_counts, ind, args.vocab_size, device)
            sums = data_batch.sum(1).unsqueeze(1)
            cnt += sums.sum(0).squeeze().cpu().numpy()
            if args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            theta, _ = model.get_theta(normalized_data_batch)
            thetaAvg += theta.sum(0).unsqueeze(0) / args.num_docs_train
            weighed_theta = sums * theta
            thetaWeightedAvg += weighed_theta.sum(0).unsqueeze(0)
            if idx % 100 == 0 and idx > 0:
                print('batch: {}/{}'.format(idx, len(indices)))
        thetaWeightedAvg = thetaWeightedAvg.squeeze().cpu().numpy() / cnt
        print('\nThe 10 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:10]))

        ## show topics
        beta = model.get_beta()
        topic_indices = list(np.random.choice(args.num_topics, 10)) # 10 random topics
        print('\n')
        for k in range(args.num_topics):#topic_indices:
            gamma = beta[k]
            top_words = list(gamma.cpu().numpy().argsort()[-args.num_words+1:][::-1])
            topic_words = [vocab[a] for a in top_words]
            print('Topic {}: {}'.format(k, topic_words))

        if args.train_embeddings:
            ## show etm embeddings 
            try:
                rho_etm = model.rho.weight.cpu()
            except:
                rho_etm = model.rho.cpu()
            queries = ['andrew', 'woman', 'computer', 'sports', 'religion', 'man', 'love', 
                            'intelligence', 'money', 'politics', 'health', 'people', 'family']
            print('\n')
            print('ETM embeddings...')
            for word in queries:
                print('word: {} .. etm neighbors: {}'.format(word, nearest_neighbors(word, rho_etm, vocab)))
            print('\n')
"""

