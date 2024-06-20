# Organizing the imports
# Standard libraries
import string
import pickle
from collections import defaultdict
import math

# Libraries for Deep Learning and ML
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy import sparse
from octis.models.vONTSS_model.hyperspherical_vae.distributions.von_mises_fisher import VonMisesFisher, HypersphericalUniform
from octis.models.model import AbstractModel
from torch.distributions.kl import register_kl


# Libraries for NLP
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import gensim.downloader
import gensim

# Other utilities
import pandas as pd
import numpy as np
# import ot
import matplotlib.pyplot as plt
# import seaborn as sns
# from datasets import Dataset
from octis.models.vONTSS_model.utils import kld_normal
from octis.models.vONTSS_model.preprocess import TextProcessor

@register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu):
    #print(vmf.entropy() , hyu.entropy())
    return -vmf.entropy()  + hyu.entropy()
 

# Libraries for NLP
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import gensim.downloader
import gensim

# Other utilities
import pandas as pd
import numpy as np
# import ot
import matplotlib.pyplot as plt
# import seaborn as sns
# from datasets import Dataset
from octis.models.vONTSS_model.utils import kld_normal
from octis.models.vONTSS_model.preprocess import TextProcessor

@register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu):
    #print(vmf.entropy() , hyu.entropy())
    return -vmf.entropy()  + hyu.entropy()


class EmbTopic(nn.Module):
    """
    A class used to represent decoder for Embedded Topic Modeling 
    reimplement of: https://github.com/lffloyd/embedded-topic-model
    
    Attributes
    ----------
    topic_emb: nn.Parameters
        represent topic embedding
    
    
    Methods:
    --------
    forward(logit)
        Output the result from decoder
    get_topics
        result before log
    
    
    """
    def __init__(self, embedding, k, normalize = False):
        super(EmbTopic, self).__init__()
        self.embedding = embedding
        n_vocab, topic_dim = embedding.weight.size()
        self.k = k
        self.topic_emb = nn.Parameter(torch.Tensor(k, topic_dim))
        self.reset_parameters()
        self.normalize = normalize

    def forward(self, logit):
        # return the log_prob of vocab distribution
#         if normalize:
#             self.topic_emb = torch.nn.Parameter(normalize(self.topic_emb))
        if self.normalize:
            val = normalize(self.topic_emb) @ self.embedding.weight.transpose(0, 1)
        else: 
            val = self.topic_emb @ self.embedding.weight.transpose(0, 1)
        # print(val.shape)
        beta = F.softmax(val, dim=1)
        # print(beta.shape)
        # return beta
        return torch.log(torch.matmul(logit, beta) + 1e-10)

    def get_topics(self):
        return F.softmax(self.topic_emb @ self.embedding.weight.transpose(0, 1), dim=1)
    
    
    def get_rank(self):
        #self.topic_emb = torch.nn.Parameter(normalize(self.topic_emb))
        return normalize(self.topic_emb) @ self.embedding.weight.transpose(0, 1)

    def reset_parameters(self):
        init.normal_(self.topic_emb)
        # init.kaiming_uniform_(self.topic_emb, a=math.sqrt(5))
        # init.normal_(self.embedding.weight, std=0.01)

    def extra_repr(self):
        k, d = self.topic_emb.size()
        return 'topic_emb: Parameter({}, {})'.format(k, d)
    


def topic_covariance_penalty(topic_emb, EPS=1e-12):
    """topic_emb: T x topic_dim."""
    #normalized the topic
    normalized_topic = topic_emb / (torch.norm(topic_emb, dim=-1, keepdim=True) + EPS)
    #get topic similarity absolute value
    cosine = (normalized_topic @ normalized_topic.transpose(0, 1)).abs()
    #average similarity
    mean = cosine.mean()
    #variance
    var = ((cosine - mean) ** 2).mean()
    return mean - var, var, mean

class NormalParameter(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormalParameter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features)
        self.log_sigma = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def forward(self, h):
        return self.mu(h), self.log_sigma(h)

    def reset_parameters(self):
        init.zeros_(self.log_sigma.weight)
        init.zeros_(self.log_sigma.bias)

class NTM(nn.Module):
    """NTM that keeps track of output
    """
    def __init__(self, hidden, normal, h_to_z, topics):
        super(NTM, self).__init__()
        self.hidden = hidden
        self.normal = normal
        self.h_to_z = h_to_z
        self.topics = topics
        self.output = None
        self.drop = nn.Dropout(p=0.5)
    def forward(self, x, n_sample=1):
        h = self.hidden(x)
        h = self.drop(h)
        mu, log_sigma = self.normal(h)
        #identify how far it is away from normal distribution
        kld = kld_normal(mu, log_sigma)
        #print(kld.shape)
        rec_loss = 0
        for i in range(n_sample):
            #reparametrician trick
            z = torch.zeros_like(mu).normal_() * torch.exp(0.5*log_sigma) + mu
            #decode
            
            z = self.h_to_z(z)
            self.output = z
            #print(z)
            #z = self.drop(z)
            #get log probability for reconstruction loss
            log_prob = self.topics(z)
            rec_loss = rec_loss - (log_prob * x).sum(dim=-1)
        #average reconstruction loss
        rec_loss = rec_loss / n_sample
        #print(rec_loss.shape)
        minus_elbo = rec_loss + kld

        return {
            'loss': minus_elbo,
            'minus_elbo': minus_elbo,
            'rec_loss': rec_loss,
            'kld': kld
        }

    def get_topics(self):
        return self.topics.get_topics()

def optimal_transport_prior(softmax_top,  index, 
                            lambda_sh = 1):
    """ add prior as a semi-supervised loss
    
    parameters
    ----------
    softmax_top: softmax results from decoder
    index: list: a list of list with number as index
    embedding: numpy array, word embedding trained by spherical word embeddings
    beta: float, weights for prior loss
    gamma: float, weights for negative sampling
    iter2: int, how many epochs to train for third phase
    sample: int, sample number
    lambda_sh: low means high entrophy
    
    Returns:
    --------
    int
        loss functions
    
    """
    
    m = - torch.log(softmax_top + 1e-12)
    loss = torch.cat([m[:, i].mean(axis = 1).reshape(1, -1) for i in index]).to(m.device) 
    #print(loss.shape)
    b = torch.ones(loss.shape[1]).to(m.device) 
    a = torch.ones(loss.shape[0]).to(m.device) 

    return ot.sinkhorn(a, b, loss, lambda_sh).sum()

class VNTM(nn.Module):
    """NTM that keeps track of output
    """
    def __init__(self, hidden, normal, h_to_z, topics, layer, top_number, penalty, beta = 1, index = None, temp=10):
        super(VNTM, self).__init__()
        self.hidden = hidden
        #self.normal = normal
        self.h_to_z = h_to_z
        self.topics = topics
        self.output = None
        self.index = index
        self.drop = nn.Dropout(p=0.3)
        self.fc_mean = nn.Linear(layer, top_number)
        self.fc_var = nn.Linear(layer, 1)
        self.num = top_number
        self.penalty = penalty
        self.temp = temp
        self.beta = beta

        #self.dirichlet = torch.distributions.dirichlet.Dirichlet((torch.ones(self.topics.k)/self.topics.k).cuda())
    def forward(self, x, device, n_sample=1, epoch = 0):
        h = self.hidden(x)
        h = self.drop(h)
        z_mean = self.fc_mean(h)
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
        # the `+ 1` prevent collapsing behaviors
        z_var = F.softplus(self.fc_var(h)) + 1
        
        q_z = VonMisesFisher(z_mean, z_var)
        p_z = HypersphericalUniform(self.num - 1, device=device)
        kld = torch.distributions.kl.kl_divergence(q_z, p_z).mean().to(device)
        #print(q_z)
        #mu, log_sigma = self.normal(h)
        #identify how far it is away from normal distribution
        
        #print(kld.shape)
        rec_loss = 0
        for i in range(n_sample):
            #reparametrician trick
            z = q_z.rsample()
            #z = nn.Softmax()(z)
            #decode
            #print(z)
            
            z = self.h_to_z(self.temp * z)
            self.output = z
            #print(z)
            
            #get log probability for reconstruction loss
            log_prob = self.topics(z)
            rec_loss = rec_loss - (log_prob * x).sum(dim=-1)
        #average reconstruction loss
        rec_loss = rec_loss / n_sample
        #print(rec_loss.shape)
        minus_elbo = rec_loss + kld
        penalty, var, mean = topic_covariance_penalty(self.topics.topic_emb) 
        if self.index is not None:
            sinkhorn = optimal_transport_prior(self.topics.get_topics(), self.index)
        else:
            sinkhorn = 0

        return {
            'loss': minus_elbo + penalty * self.penalty + sinkhorn * self.beta,
            'minus_elbo': minus_elbo,
            'rec_loss': rec_loss,
            'kld': kld
        }

    def get_topics(self):
        return self.topics.get_topics()

def get_mlp(features, activate):
    """features: mlp size of each layer, append activation in each layer except for the first layer."""
    if isinstance(activate, str):
        activate = getattr(nn, activate)
    layers = []
    for in_f, out_f in zip(features[:-1], features[1:]):
        layers.append(nn.Linear(in_f, out_f))
        layers.append(activate())
    return nn.Sequential(*layers)

class GSM(NTM):
    def __init__(self, hidden, normal, h_to_z, topics, penalty):
        # h_to_z will output probabilities over topics
        super(GSM, self).__init__(hidden, normal, h_to_z, topics)
        self.penalty = penalty

    def forward(self, x, device, n_sample=1):
        stat = super(GSM, self).forward(x, n_sample)
        loss = stat['loss'].to(device)
        penalty, var, mean = topic_covariance_penalty(self.topics.topic_emb)

        stat.update({
            'loss': loss #+ penalty.to(device) * self.penalty,
            # 'penalty_mean': mean,
            # 'penalty_var': var,
            # 'penalty': penalty.to(device) * self.penalty,
        })

        return stat
    
class Topics(nn.Module):
    def __init__(self, k, vocab_size, bias=True):
        super(Topics, self).__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.topic = nn.Linear(k, vocab_size, bias=bias)

    def forward(self, logit):
        # return the log_prob of vocab distribution
        return torch.log_softmax(self.topic(logit), dim=-1)

    def get_topics(self):
        return torch.softmax(self.topic.weight.data.transpose(0, 1), dim=-1)

    def get_topic_word_logit(self):
        """topic x V.
        Return the logits instead of probability distribution
        """
        return self.topic.weight.transpose(0, 1)


class VONT(AbstractModel):
    def __init__(self, epochs=20, batch_size=256, gpu_num=1, numb_embeddings=20, 
                 learning_rate=0.002, weight_decay=1.2e-6, penalty=1, beta = 1, temp = 10,
                 top_n_words=20, num_representative_docs=5, top_n_topics=100, embedding_dim=100):

        self.dataset = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu_num = gpu_num
        self.numb_embeddings = numb_embeddings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.penalty = penalty
        self.top_n_words = top_n_words
        self.num_representative_docs = num_representative_docs
        self.top_n_topics = top_n_topics
        self.embedding_dim = embedding_dim
        self.device = torch.device("cpu")
        self.beta = beta
        self.temp = temp
        self.z = None
        self.model = None

    def train(self, X, batch_size):
        self.model.train()
        total_nll = 0.0
        total_kld = 0.0

        indices = torch.randperm(X.shape[0])
        indices = torch.split(indices, batch_size)
        length = len(indices)
        for idx, ind in enumerate(indices):
            data_batch = X[ind].to(self.device).float()

            d = self.model(x = data_batch, device = self.device)

            total_nll += d['rec_loss'].sum().item() / batch_size
            total_kld += d['kld'].sum().item() / batch_size  
            loss = d['loss']

            self.optimizer.zero_grad()
            loss.sum().backward()
            self.optimizer.step()
            self.scheduler.step()

        print(total_nll/length, total_kld/length)

    def fit_transform(self, dataset, index = []):
        self.dataset = dataset
        self.tp = TextProcessor(self.dataset)
        self.tp.process()
        bag_of_words = torch.tensor(self.tp.bow)
        if index != []:
            index_words = [[self.tp.word_to_index[word] for word in ind if word in self.tp.word_to_index] for ind in index]
        else:
            index_words = None
        print(index_words)
        #print(bag_of_words.shape)
        # rest of your initialization code here
        layer = bag_of_words.shape[1]//16
        hidden = get_mlp([bag_of_words.shape[1], bag_of_words.shape[1]//4, layer], nn.GELU)
        normal = NormalParameter(layer, self.numb_embeddings)
        h_to_z = nn.Softmax()
        embedding = nn.Embedding(bag_of_words.shape[1], 100)
        # p1d = (0, 0, 0, 10000 - company1.embeddings.shape[0]) # pad last dim by 1 on each side
        # out = F.pad(company1.embeddings, p1d, "constant", 0)  # effectively zero padding

        glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
        embed = np.asarray([glove_vectors[self.tp.index_to_word[i]] if  self.tp.index_to_word[i] in glove_vectors else np.asarray([1]*100) for i in self.tp.index_to_word ])
        print(embed.shape)
        embedding.weight = torch.nn.Parameter(torch.from_numpy(embed).float())
        embedding.weight.requires_grad=True


       
        topics = EmbTopic(embedding = embedding,
                            k = self.numb_embeddings, normalize = False)


        
        
        self.model = VNTM(hidden = hidden,
                    normal = normal,
                    h_to_z = h_to_z,
                    topics = topics,
                    layer = layer, 
                    top_number = self.numb_embeddings,
                    index = index_words, 
                    penalty = self.penalty,
                    beta = self.beta,
                    temp = self.temp,
                    ).to(self.device).float()

        #batch_size = 256
        self.optimizer = optim.Adam(self.model.parameters(), 
                                lr=self.learning_rate, 
                                weight_decay=self.weight_decay)




        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.002, steps_per_epoch=int(bag_of_words.shape[0]/self.batch_size) + 1, epochs=self.epochs)

        # Initialize and train your model
        for epoch in range(self.epochs):
            self.train(bag_of_words, self.batch_size)
        
        # Store the topics
        emb = self.model.topics.get_topics().cpu().detach().numpy()
        self.topics =  [[self.tp.index_to_word[ind] for ind in np.argsort(emb[i])[::-1][:self.top_n_topics]] for i in range(self.numb_embeddings)] #100 can be specified
        self.topics_score = [[score for score in np.sort(emb[i])[::-1]] for i in range(self.numb_embeddings)] 
        # Compute and store the documents-topics distributions
        data_batch = bag_of_words.float()
        self.model.cpu()

        z = self.model.hidden(data_batch)
        z_mean = self.model.fc_mean(z)
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
        self.z = self.model.h_to_z(z_mean).detach().numpy()
        self.topic_doc =  [[ind for ind in np.argsort(self.z[:, i])[::-1][:100] ] for i in range(self.numb_embeddings)] #100 can be specified
        self.topic_doc_score = [[ind for ind in np.sort(self.z[:, i])[::-1][:100] ] for i in range(self.numb_embeddings)] #100 can be specified

    
        return self.topics, self.z
    
    def get_topics(self, index):
        return  [(i, j) for i, j in zip(self.topics[index], self.topics_score[index])][:self.top_n_words]

    def get_representative_docs(self, index):
        return  [(self.dataset[i], j) for i, j in zip(self.topic_doc[index], self.topic_doc_score[index])][:self.num_representative_docs]

    def topic_word_matrix(self):
        return self.model.topics.get_topics().cpu().detach().numpy()

    def topic_keywords(self):
        return self.topics
    
#     def visualize_topic_similarity(self):
#         # Compute m similarity matrix
#         topic_word_matrix = self.model.topics.topic_emb.detach().numpy()
#         similarity_matrix = np.matmul(topic_word_matrix, topic_word_matrix.T)

#         # Plot the similarity matrix as a heatmap
#         plt.figure(figsize=(10, 10))
#         sns.heatmap(similarity_matrix, cmap="YlGnBu", square=True)
#         plt.title('Topic Similarity Heatmap')
#         plt.xlabel('Topic IDs')
#         plt.ylabel('Topic IDs')
#         plt.show()

    def visualize_topic_keywords(self, topic_id, num_keywords=10):
        # Get top keywords for the given topic
        topic_keywords = self.get_topics(topic_id)[:num_keywords]
        words, scores = zip(*topic_keywords)

        # Generate the bar plot
        plt.figure(figsize=(10, 5))
        plt.barh(words, scores, color='skyblue')
        plt.xlabel("Keyword Importance")
        plt.title(f"Top {num_keywords} Keywords for Topic {topic_id}")
        plt.gca().invert_yaxis()
        plt.show()

    def get_document_info(self, top_n_words=10):
        data = []
        for topic_id in range(self.numb_embeddings):
            topic_keywords = self.get_topics(topic_id)[:top_n_words]
            topic_keywords_str = "_".join([word for word, _ in topic_keywords[:3]])

            # Get the document that has the highest probability for this topic
            doc_indices = np.argsort(self.z[:, topic_id])[::-1]
            representative_doc_index = doc_indices[0]
            representative_doc = self.dataset[representative_doc_index]

            # Count the number of documents that have this topic as their dominant topic
            dominant_topics = np.argmax(self.z, axis=1)
            num_docs = np.sum(dominant_topics == topic_id)

            data.append([topic_id, f"{topic_id}_{topic_keywords_str}", topic_keywords_str, representative_doc, num_docs])

        df = pd.DataFrame(data, columns=["Topic", "Name", "Top_n_words", "Representative_Doc", "Num_Docs"])
        return df
    
    def train_model(self, dataset, hyperparameters={}, top_words=10):
        
        self.top_n_words = top_words    
        # Extract hyperparameters and set them as attributes
        if 'epochs' in hyperparameters:
            self.epochs = hyperparameters['epochs']
        if 'batch_size' in hyperparameters:
            self.batch_size = hyperparameters['batch_size']
        if 'gpu_num' in hyperparameters:
            self.gpu_num = hyperparameters['gpu_num']
        if 'numb_embeddings' in hyperparameters:
            self.numb_embeddings = hyperparameters['numb_embeddings']
        if 'learning_rate' in hyperparameters:
            self.learning_rate = hyperparameters['learning_rate']
        if 'weight_decay' in hyperparameters:
            self.weight_decay = hyperparameters['weight_decay']
        if 'penalty' in hyperparameters:
            self.penalty = hyperparameters['penalty']
        if 'beta' in hyperparameters:
            self.beta = hyperparameters['beta']
        if 'temp' in hyperparameters:
            self.temp = hyperparameters['temp']
        
        if 'num_representative_docs' in hyperparameters:
            self.num_representative_docs = hyperparameters['num_representative_docs']
        if 'top_n_topics' in hyperparameters:
            self.top_n_topics = hyperparameters['top_n_topics']
        if 'embedding_dim' in hyperparameters:
            self.embedding_dim = hyperparameters['embedding_dim']

        # Check if the model has been trained
        if self.z is None:
            self.fit_transform(dataset)

        # Create the model output
        model_output = {}
        model_output['topics'] = [i[:top_words] for i in self.topics]
        model_output['topic-word-matrix'] = self.model.topics.get_topics().cpu().detach().numpy()
        model_output['topic-document-matrix'] = self.z.T

        return model_output

