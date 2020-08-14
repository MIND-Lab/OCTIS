from __future__ import print_function
from models.early_stopping.pytorchtools import EarlyStopping

import argparse
import torch
import pickle
import numpy as np
from models.ETM import data
from sklearn.feature_extraction.text import CountVectorizer
from torch import nn, optim
from models.ETM import etm
from models.model import Abstract_Model


class ETM_Wrapper(Abstract_Model):

    def __init__(self):
        self.hyperparameters = {}
        self.bool_topic_doc = True
        self.bool_topic_word = True
        self.top_word = 10
        self.early_stopping = EarlyStopping(patience=5, verbose=True)


    def train_model(self, dataset, hyperparameters, top_words=10, topic_word_matrix=True,
                    topic_document_matrix=True, embeddings=None, train_embeddings=True):
        print(train_embeddings, "pre_set_model")
        self.set_model(dataset, hyperparameters, embeddings, train_embeddings)
        print(train_embeddings, "post_set_model")
        self.bool_topic_doc = topic_document_matrix
        self.bool_topic_word = topic_word_matrix
        self.top_word = top_words
####################################
        for epoch in range(0, self.hyperparameters['num_epochs']):
            continue_training = self._train_epoch(epoch)
            if not continue_training:
                break

        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load('checkpoint.pt'))

        if self.use_partitions:
            result = self.inference()
        else:
            result = self.get_info()

        return result

    def set_model(self, dataset, hyperparameters, embeddings, train_embeddings):
        if self.use_partitions:
            train_data, validation_data, testing_data = \
                dataset.get_partitioned_corpus(use_validation=True)

            data_corpus_train = [' '.join(i) for i in train_data]
            data_corpus_test = [' '.join(i) for i in testing_data]
            data_corpus_val = [' '.join(i) for i in validation_data]

            self.train_tokens, self.train_counts, self.test_tokens, \
            self.test_counts, self.valid_tokens, self.valid_counts, self.vocab_size, \
            self.vocab = self.preprocess(data_corpus_train, data_corpus_test, data_corpus_val)
        else:
            data_corpus = [' '.join(i) for i in dataset.get_corpus()]
            self.train_tokens, self.train_counts, self.vocab_size, self.vocab = self.preprocess(data_corpus, None)

        self.num_docs_train = self.train_tokens.shape[1]
        self.num_docs_valid = self.valid_tokens.shape[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #np.random.seed(0)
        #torch.manual_seed(0)
        #if torch.cuda.is_available():
        #    torch.cuda.manual_seed(0)

        self.set_default_hyperparameters(hyperparameters)
        ## define model and optimizer
        self.model = etm.ETM(num_topics=self.hyperparameters['num_topics'],
                             vocab_size=self.vocab_size,
                             t_hidden_size=self.hyperparameters['t_hidden_size'],
                             rho_size=self.hyperparameters['rho_size'],
                             emsize=self.hyperparameters['emb_size'],
                             theta_act=self.hyperparameters['theta_act'],
                             embeddings=embeddings, train_embeddings=train_embeddings,
                             enc_drop=self.hyperparameters['enc_drop']).to(self.device)
        print('model: {}'.format(self.model))

        self.optimizer = self.set_optimizer()

    def set_optimizer(self):
        if self.hyperparameters['optimizer'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparameters['lr'])
        elif self.hyperparameters['optimizer'] == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.hyperparameters['lr'])
        elif self.hyperparameters['optimizer'] == 'adadelta':
            optimizer = optim.Adadelta(self.model.parameters(), lr=self.hyperparameters['lr'])
        elif self.hyperparameters['optimizer'] == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.hyperparameters['lr'])
        elif self.hyperparameters['optimizer'] == 'asgd':
            optimizer = optim.ASGD(self.model.parameters(), lr=self.hyperparameters['lr'],
                                   t0=0, lambd=0.)
        else:
            print('Defaulting to vanilla SGD')
            optimizer = optim.SGD(self.model.parameters(), lr=self.hyperparameters['lr'])

        return optimizer

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
            data_batch = data.get_batch(self.train_tokens, self.train_counts,
                                        ind, self.vocab_size,
                                        self.hyperparameters['emb_size'], self.device)
            sums = data_batch.sum(1).unsqueeze(1)
            if self.hyperparameters['bow_norm']:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            recon_loss, kld_theta = self.model(data_batch, normalized_data_batch)
            total_loss = recon_loss + kld_theta
            total_loss.backward()
            if self.hyperparameters["clip"] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.hyperparameters["clip"])
            self.optimizer.step()

            acc_loss += torch.sum(recon_loss).item()
            acc_kl_theta_loss += torch.sum(kld_theta).item()
            cnt += 1
            log_interval = 20
            if idx % log_interval == 0 and idx > 0:
                cur_loss = round(acc_loss / cnt, 2)
                cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
                cur_real_loss = round(cur_loss + cur_kl_theta, 2)

                print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {}'
                      ' .. NELBO: {}'.format(epoch, idx, len(indices),
                                             self.optimizer.param_groups[0]['lr'],
                                             cur_kl_theta, cur_loss, cur_real_loss))

            self.data_list.append(normalized_data_batch)

        cur_loss = round(acc_loss / cnt, 2)
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
        cur_real_loss = round(cur_loss + cur_kl_theta, 2)
        print('*' * 100)
        print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
            epoch + 1, self.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss,
            cur_real_loss))
        print('*' * 100)

        # VALIDATION ###
        model = self.model.to(self.device)
        model.eval()
        with torch.no_grad():
            val_acc_loss = 0
            val_acc_kl_theta_loss = 0
            val_cnt = 0
            indices = torch.arange(0, self.num_docs_valid)
            indices = torch.split(indices, self.hyperparameters['batch_size'])
            for idx, ind in enumerate(indices):
                self.optimizer.zero_grad()
                self.model.zero_grad()
                val_data_batch = data.get_batch(self.valid_tokens, self.valid_counts,
                                                ind, self.vocab_size,
                                                self.hyperparameters['emb_size'], self.device)
                sums = val_data_batch.sum(1).unsqueeze(1)
                if self.hyperparameters['bow_norm']:
                    val_normalized_data_batch = val_data_batch / sums
                else:
                    val_normalized_data_batch = val_data_batch

                val_recon_loss, val_kld_theta = self.model(val_data_batch,
                                                           val_normalized_data_batch)
                val_total_loss = val_recon_loss + val_kld_theta

                val_acc_loss += torch.sum(val_recon_loss).item()
                val_acc_kl_theta_loss += torch.sum(val_kld_theta).item()
                val_cnt += 1

            val_cur_loss = round(val_acc_loss / cnt, 2)
            val_cur_kl_theta = round(val_acc_kl_theta_loss / cnt, 2)
            val_cur_real_loss = round(val_cur_loss + val_cur_kl_theta, 2)
            print('*' * 100)
            print('VALIDATION .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                self.optimizer.param_groups[0]['lr'], val_cur_kl_theta, val_cur_loss,
                val_cur_real_loss))
            print('*' * 100)

            self.early_stopping(val_cur_real_loss, model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                return False
            else:
                return True

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
                # print('Topic {}: {}'.format(k, topic_words))
                topic_w.append(topic_words)
        if self.bool_topic_doc and self.bool_topic_word:
            info['topics'] = topic_w
            info['topic-word-matrix'] = self.model.get_beta().cpu().detach().numpy()
            info['topic-document-matrix'] = theta.cpu().detach().numpy()
        elif self.bool_topic_doc and not self.bool_topic_word:
            info['topics'] = topic_w
            info['topic-document-matrix'] = theta.cpu().detach().numpy()
        elif not self.bool_topic_doc and self.bool_topic_word:
            info['topics'] = topic_w
            info['topic-word-matrix'] = self.model.get_beta().cpu().detach().numpy()
        else:
            info['topics'] = topic_w
        return info

    def inference(self):
        assert isinstance(self.use_partitions, bool) and self.use_partitions
        topic_d = []
        self.model.eval()
        indices = torch.arange(0, self.test_tokens.shape[1])
        indices = torch.split(indices, self.hyperparameters['batch_size'])

        for idx, ind in enumerate(indices):
            data_batch = data.get_batch(self.test_tokens, self.test_counts,
                                        ind, self.vocab_size,
                                        self.hyperparameters['emb_size'], self.device)
            sums = data_batch.sum(1).unsqueeze(1)
            if self.hyperparameters['bow_norm']:
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


    #def get_test(self):
    #    tok = self.test_tokens
    #    cou = self.test_counts
    #    return tok, cou

    def set_default_hyperparameters(self, hyperparameters):
        self.hyperparameters['num_topics'] = hyperparameters.get(
            'num_topics', self.hyperparameters.get('num_topics', 10))
        self.hyperparameters['num_epochs'] = hyperparameters.get(
            'num_epochs', self.hyperparameters.get('num_epochs', 20))
        self.hyperparameters['t_hidden_size'] = hyperparameters.get(
            't_hidden_size', self.hyperparameters.get('t_hidden_size', 800))
        self.hyperparameters['rho_size'] = hyperparameters.get(
            'rho_size', self.hyperparameters.get('rho_size', 300))
        self.hyperparameters['emb_size'] = hyperparameters.get(
            'emb_size', self.hyperparameters.get('emb_size', 300))
        self.hyperparameters['theta_act'] = hyperparameters.get(
            'theta_act', self.hyperparameters.get('theta_act', 'relu'))
        self.hyperparameters['enc_drop'] = hyperparameters.get(
            'enc_drop', self.hyperparameters.get('enc_drop', 0.0))
        self.hyperparameters['lr'] = hyperparameters.get(
            'lr', self.hyperparameters.get('lr', 0.005))
        self.hyperparameters['optimizer'] = hyperparameters.get(
            'optimizer', self.hyperparameters.get('optimizer', 'adam'))
        self.hyperparameters['batch_size'] = hyperparameters.get(
            'batch_size', self.hyperparameters.get('batch_size', 128))
        self.hyperparameters['clip'] = hyperparameters.get(
            'clip', self.hyperparameters.get('clip', 0.0))
        self.hyperparameters['wdecay'] = hyperparameters.get(
            'wdecay', self.hyperparameters.get('wdecay', 1.2e-6))
        self.hyperparameters['anneal_lr'] = hyperparameters.get(
            'anneal_lr', self.hyperparameters.get('anneal_lr', 0))
        self.hyperparameters['bow_norm'] = hyperparameters.get(
            'bow_norm', self.hyperparameters.get('bow_norm', 1))


    def partitioning(self, use_partitions=False):
        self.use_partitions = use_partitions

    @staticmethod
    def preprocess(train_corpus, test_corpus=None, validation_corpus=None):
        def split_bow(bow_in, n_docs):
            indices = np.asarray([np.asarray([w for w in bow_in[doc, :].indices]) for doc in range(n_docs)])
            counts = np.asarray([np.asarray([c for c in bow_in[doc, :].data]) for doc in range(n_docs)])
            return np.expand_dims(indices, axis=0), np.expand_dims(counts, axis=0)

        vec = CountVectorizer(token_pattern=r'(?u)\b\w+\b')

        dataset = train_corpus
        if test_corpus is not None:
            dataset.extend(test_corpus)
        if validation_corpus is not None:
            dataset.extend(validation_corpus)

        vec.fit(dataset)
        idx2token = {v: k for (k, v) in vec.vocabulary_.items()}
        vocab_size = len(idx2token.keys())
        vocab = vec.vocabulary_.keys()

        X_train = vec.transform(train_corpus)
        X_train_tokens, X_train_count = split_bow(X_train, X_train.shape[0])

        if test_corpus is not None:
            X_test = vec.transform(test_corpus)
            X_test_tokens, X_test_count = split_bow(X_test, X_test.shape[0])

        if validation_corpus is not None:
            X_validation = vec.transform(validation_corpus)
            X_val_tokens, X_val_count = split_bow(X_validation, X_validation.shape[0])

        if test_corpus is not None and validation_corpus is not None:
            return X_train_tokens, X_train_count, X_test_tokens, X_test_count, X_val_tokens, X_val_count, vocab_size, list(vocab)
        elif test_corpus is not None and validation_corpus is None:
            return X_train_tokens, X_train_count, X_test_tokens, X_test_count, vocab_size, list(vocab)
        elif test_corpus is None and validation_corpus is not None:
            return X_train_tokens, X_train_count, X_val_tokens, X_val_count, vocab_size, list(vocab)
        elif test_corpus is None and validation_corpus is None:
            return X_train_tokens, X_train_count, vocab_size, list(vocab)
        else:
            print("something strange is happening?")


