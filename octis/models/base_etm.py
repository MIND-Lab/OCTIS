from octis.models.model import AbstractModel
import pickle as pkl
import numpy as np
from torch import nn, optim
import gensim
import torch

class BaseETM(AbstractModel):
    """
    this is the base model both the embedde
    and the dynamic embedded topic model will inherit from
    it since it  contains the methods that are share among the both models
    """
    def set_optimizer(self):
        self.hyperparameters['lr'] = float(self.hyperparameters['lr'])
        self.hyperparameters['wdecay'] = float(self.hyperparameters['wdecay'])
        if self.hyperparameters['optimizer'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparameters['lr'],
                                   weight_decay=self.hyperparameters['wdecay'])
        elif self.hyperparameters['optimizer'] == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.hyperparameters['lr'],
                                      weight_decay=self.hyperparameters['wdecay'])
        elif self.hyperparameters['optimizer'] == 'adadelta':
            optimizer = optim.Adadelta(self.model.parameters(), lr=self.hyperparameters['lr'],
                                       weight_decay=self.hyperparameters['wdecay'])
        elif self.hyperparameters['optimizer'] == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.hyperparameters['lr'],
                                      weight_decay=self.hyperparameters['wdecay'])
        elif self.hyperparameters['optimizer'] == 'asgd':
            optimizer = optim.ASGD(self.model.parameters(), lr=self.hyperparameters['lr'],
                                   t0=0, lambd=0., weight_decay=self.hyperparameters['wdecay'])
        elif self.hyperparameters['optimizer'] == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.hyperparameters['lr'],
                                  weight_decay=self.hyperparameters['wdecay'])
        else:
            print('Defaulting to vanilla SGD')
            optimizer = optim.SGD(self.model.parameters(), lr=self.hyperparameters['lr'])

        return optimizer


    @staticmethod
    def preprocess(vocab2id, train_corpus, test_corpus=None, validation_corpus=None):

        raise NotImplementedError("Subclasses should implement this!")

    def load_embeddings(self):
        if not self.hyperparameters['train_embeddings']:
            vectors = {}
            embs = pkl.load(open(self.hyperparameters['embeddings_path'], 'rb'))
            for l in embs:
                line = l.split()
                word = line[0]
                if word in self.vocab.values():
                    vect = np.array(line[1:]).astype(np.float)
                    vectors[word] = vect
            embeddings = np.zeros((len(self.vocab.keys()), self.hyperparameters['embedding_size']))
            words_found = 0
            for i, word in enumerate(self.vocab.values()):
                try:
                    embeddings[i] = vectors[word]
                    words_found += 1
                except KeyError:
                    embeddings[i] = np.random.normal(scale=0.6, size=(self.hyperparameters['embedding_size'],))
            self.embeddings = torch.from_numpy(embeddings).to(self.device)

    def filter_pretrained_embeddings(self, pretrained_embeddings_path, save_embedding_path, vocab_path, binary=True):
        """
        Filter the embeddings from a set of word2vec-format pretrained embeddings based on the vocabulary
        This should allow you to avoid to load the whole embedding space every time you do Bayesian Optimization
        but just the embeddings that are in the vocabulary.
        :param pretrained_embeddings_path:
        :return:
        """
        vocab = []
        with open(vocab_path, 'r') as fr:
            for line in fr.readlines():
                vocab.append(line.strip().split(" ")[0])

        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=binary)
        embeddings = []
        for word in vocab:
            if word in w2v_model.vocab:
                line = word
                for w in w2v_model[word].tolist():
                    line = line + " " + str(w)
                embeddings.append(line)
        pkl.dump(embeddings, open(save_embedding_path, 'wb'))
