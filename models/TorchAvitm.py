#!/usr/bin/env python
# coding: utf-8

# In[1]:

from models.model import Abstract_Model
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 
import numpy as np 

from models.pytorchavitm.avitm import avitm
from models.pytorchavitm import datasets


class TorchAvitm(Abstract_Model):
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

    def train_model(self, dataset, hyperparameters, top_words=10, topic_word_matrix=True, topic_document_matrix=True):

        self.n_components = hyperparameters.get('n_components', 10)
        self.model_type = hyperparameters.get('model_type', 'prodLDA')
        self.hidden_sizes = hyperparameters.get('hidden_sizes', (100,100))
        self.activation = hyperparameters.get('activation', 'softplus')
        self.dropout = hyperparameters.get('dropout', 0.2)
        self.learn_priors = hyperparameters.get('learn_priors', True)
        self.batch_size = hyperparameters.get('batch_size', 64)
        self.lr = hyperparameters.get('lr', 2e-3)
        self.momentum = hyperparameters.get('momentum', 0.99)
        self.solver = hyperparameters.get('solver', 'adam')
        self.num_epochs = hyperparameters.get('num_epochs', 100)
        self.reduce_on_plateau = hyperparameters.get('reduce_on_plateau', False)
        
        data_corpus = [','.join(i) for i in dataset.get_corpus()]
        
        self.X_train, self.input_size = self.preprocess(data_corpus)
      
        self.avitm_model = avitm.AVITM(input_size=self.input_size, n_components=self.n_components,
                                  model_type=self.model_type,hidden_sizes=self.hidden_sizes, 
                                  activation=self.activation, dropout=self.dropout,
                                  learn_priors=self.learn_priors, batch_size=self.batch_size, 
                                  lr=self.lr, momentum=self.momentum,
                                  solver=self.solver, num_epochs=self.num_epochs, 
                                  reduce_on_plateau=self.reduce_on_plateau)
    
        self.avitm_model.fit(self.X_train)
        result = self.avitm_model.get_info()
        
        return result 

    
    @staticmethod
    def preprocess(data):
        
        def to_bow(data, min_length):
            """Convert index lists to bag of words representation of documents."""
            vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length)
                    for x in data if np.sum(x[x != np.array(None)]) != 0]
            return np.array(vect)
        
        vec = CountVectorizer()
        X = vec.fit_transform(data)
        idx2token = {v: k for (k, v) in vec.vocabulary_.items()} 
        #train_data = datasets.BOWDataset(X.toarray(), idx2token)
        train_bow = to_bow(X.toarray(), len(idx2token.keys()))
        train_data = datasets.BOWDataset(train_bow, idx2token)
        input_size = len(idx2token.keys())
        
        return train_data, input_size 
    



