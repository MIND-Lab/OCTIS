from octis.models.model import AbstractModel
import pickle as pkl
import numpy as np
from torch import optim
from gensim.models import KeyedVectors
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
        if self.hyperparameters['train_embeddings']:
            return

        vectors = self._load_word_vectors(self.hyperparameters['embeddings_path'],
                                        self.hyperparameters['embeddings_type'],
                                        self.hyperparameters['binary_embeddings'],
                                        self.hyperparameters['headerless_embeddings'])
        embeddings = np.zeros((len(self.vocab.keys()), self.hyperparameters['embedding_size']))
        for i, word in enumerate(self.vocab.values()):
            try:
                embeddings[i] = vectors[word]
            except KeyError:
                embeddings[i] = np.random.normal(scale=0.6, size=(self.hyperparameters['embedding_size'],))
        self.embeddings = torch.from_numpy(embeddings).to(self.device)

    def _load_word_vectors(self, embeddings_path, embeddings_type, binary_embeddings=True, headerless_embeddings=False):
        """
        Reads word embeddings from a specified file and format.

        :param embeddings_path: string, path to embeddings file. Can be a binary file for
            the 'pickle', 'keyedvectors' and 'word2vec' types or a text file for 'word2vec'
        :param embeddings_type: string, defines the format of the embeddings file.
            Possible values are 'pickle', 'keyedvectors' or 'word2vec'. If set to 'pickle',
            you must provide a file created with 'pickle' containing an array of word 
            embeddings, composed by words and their respective vectors. If set to 'keyedvectors', 
            you must provide a file containing a saved gensim.models.KeyedVectors instance. 
            If set to 'word2vec', you must provide a file with the original word2vec format
        :param binary_embeddings: bool, indicates if the original word2vec embeddings file is binary
            or textual (default True)
        :param headerless_embeddings: bool, indicates if the original word2vec embeddings textual file 
            has a header line in the format "<no_of_vectors> <vector_length>" (default False)
        :returns: gensim.models.KeyedVectors or dict
        """
        if embeddings_type == 'keyedvectors':
            return KeyedVectors.load(embeddings_path, mmap='r')
        elif embeddings_type == 'word2vec':
            return KeyedVectors.load_word2vec_format(
                embeddings_path, 
                binary=binary_embeddings,
                no_header=headerless_embeddings)

        vectors = {}
        embs = pkl.load(open(embeddings_path, 'rb'))
        for l in embs:
            line = l.split()
            word = line[0]
            if word in self.vocab.values():
                vect = np.array(line[1:]).astype(np.float)
                vectors[word] = vect
        return vectors

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

        w2v_model = KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=binary)
        embeddings = []
        for word in vocab:
            if word in w2v_model.vocab:
                line = word
                for w in w2v_model[word].tolist():
                    line = line + " " + str(w)
                embeddings.append(line)
        pkl.dump(embeddings, open(save_embedding_path, 'wb'))
