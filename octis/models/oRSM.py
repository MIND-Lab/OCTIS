from octis.models.model import AbstractModel
import numpy as np
from tqdm import tqdm
import gensim.corpora as corpora
import octis.configuration.citations as citations
import octis.configuration.defaults as defaults
import time

################## oRSM octis class

class oRSM(AbstractModel):

    id2word = None
    id_corpus = None
    use_partitions = True
    update_with_test = False

    def __init__(
            self, num_topics=50, epochs=5, btsz=100, M = 30, 
            lr=0.01, momentum=0.1, softstart=0.001, epsilon=0.01,
            decay=0, penalty_L1=False, penalty_local=False,
            epochs_per_monitor=1, monitor_time = False, monitor_ppl = False,
            cd_type='mfcd', K=1,
            train_optimizer='sgd', 
            logdtm=False,
            random_state=None, pretrain_epochs=1):
        
        super().__init__()
        self.hyperparameters = dict()
        self.hyperparameters["num_topics"] = num_topics
        self.hyperparameters["btsz"] = btsz
        self.hyperparameters["lr"] = lr
        self.hyperparameters["epsilon"] = epsilon
        self.hyperparameters["momentum"] = momentum
        self.hyperparameters["M"] = M
        self.hyperparameters["K"] = K
        self.hyperparameters["softstart"] = softstart
        self.hyperparameters["epochs"] = epochs
        self.hyperparameters["monitor_time"] = monitor_time
        self.hyperparameters["monitor_ppl"] = monitor_ppl
        self.hyperparameters["epochs_per_monitor"] = epochs_per_monitor
        self.hyperparameters["penalty_L1"] = penalty_L1
        self.hyperparameters["penalty_local"] = penalty_local
        self.hyperparameters["decay"] = decay
        self.hyperparameters["random_state"] = random_state
        self.hyperparameters["cd_type"] = cd_type
        self.hyperparameters["logdtm"] = logdtm
        self.hyperparameters["val_dtm"] = None
        self.hyperparameters["train_optimizer"] = train_optimizer
        self.hyperparameters['rms_decay'] = 0.9
        self.hyperparameters['adam_decay1'] = 0.9
        self.hyperparameters['adam_decay2'] = 0.999
        self.hyperparameters['pretrain_epochs'] = pretrain_epochs


    def info(self):
        """
        Returns model informations
        """
        return {
            "citation": citations.models_oRSM,
            "name": "oRSM, Over Replicated Softmax Model",
        }

    def hyperparameters_info(self):
        """
        Returns hyperparameters informations
        """
        return defaults.oRSM_hyperparameters_info

    def train_model(self, dataset, hyperparams=None, top_words=10):
        """
        Train the model and return output

        Parameters
        ----------
        dataset : dataset to use to build the model
        hyperparams : hyperparameters to build the model
        top_words : if greater than 0 returns the most significant words for
                    each topic in the output (Default True)
        Returns
        -------
        result : dictionary with up to 3 entries,
                 'topics', 'topic-word-matrix' and
                 'topic-document-matrix'
        """

        if hyperparams is None:
            hyperparams = {}

        if self.use_partitions:
            train_corpus, test_corpus = dataset.get_partitioned_corpus(use_validation = False)
        else:
            train_corpus = dataset.get_corpus()

        if self.id2word is None:
            self.id2word = self.get_vocab(dataset.get_corpus())

        if self.use_partitions:
            train_dtm = self.build_dtm(train_corpus, self.id2word)
            test_dtm = self.build_dtm(test_corpus, self.id2word)
            hyperparams["dtm"] = train_dtm
            hyperparams["val_dtm"] = test_dtm
        else:
            train_dtm = self.build_dtm(train_corpus, self.id2word)
            hyperparams["dtm"] = train_dtm
            hyperparams["val_dtm"] = None

        if "num_topics" not in hyperparams:
            hyperparams["num_topics"] = self.hyperparameters["num_topics"]

        self.hyperparameters.update(hyperparams)

        self.trained_model = oRSM_model()
        self.trained_model.train(**self.hyperparameters)

        result = {}

        result["topic-word-matrix"] = self.trained_model._get_topic_word_matrix()

        if top_words > 0:
            topics_output = []
            for topic in result["topic-word-matrix"]:
                top_k = np.argsort(topic)[-top_words:]
                top_k_words = list(reversed([self.id2word[i] for i in top_k]))
                topics_output.append(top_k_words)
            result["topics"] = topics_output

        #result["topics"] = self.trained_model.topic_words(topk=top_words, id2word=self.id2word)

        result["topic-document-matrix"] = self.trained_model.v_to_mf_h1(train_dtm).T

        if self.use_partitions:
            result["test-topic-document-matrix"] = self.trained_model.v_to_mf_h1(test_dtm).T
        else:
            result["test-topic-document-matrix"] = result["topic-document-matrix"]

        return result



############### preprocessing functions


    def get_vocab(self,tokenized_corpus):
        id2word = corpora.Dictionary(tokenized_corpus)
        return id2word


    def build_dtm(self, tokenized_corpus, id2word = None):
        """
        converts a tokenized corpus to a DOcument Term Matrix. id2word is a gensim dictionary.
        """
        if (id2word == None):
            id2word = corpora.Dictionary(tokenized_corpus)
        else:
            id2word = id2word
        id_corpus = [id2word.doc2bow(document) for document in tokenized_corpus]
        vocab = id2word.token2id
        N = len(id_corpus)
        DTM = np.zeros((N, len(vocab)))
        for i in tqdm(range(N)):
            doc = id_corpus[i]
            for id, count in doc:
                DTM[i,id] = count
        return DTM









class oRSM_model(object):

    def __init__(self):
        self.W = None

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def softmax(self, x):
        maxs = np.max(x, axis=1, keepdims=True)
        lse = maxs + np.log(np.sum(np.exp(x - maxs), axis=1, keepdims=True))
        return np.exp(x - lse)

    def multinomial_sample(self, probs, N):
        return np.random.multinomial(N, probs, size=1)[0]

    def h1_to_softmax(self, h1):
        '''
        D: number of words in the document
        h1: N x F in [0,1]
        '''
        w_vh, w_v, w_h = self.W
        energy = np.reshape(w_v, (-1,1)) + w_vh @ h1.T
        probs = self.softmax(energy.T)        
        return probs

    def sample_softmax(self, visible_probs, D):
        '''
        D: number of words in the document, for N documents
        visible_probs: N x K
        '''
        visible_sample = np.empty(visible_probs.shape)
        for i in range(visible_probs.shape[0]):
            visible_sample[i] = self.multinomial_sample(visible_probs[i], D[i])
        return visible_sample

    def sample_visible(self, h1, D):
        visible_probs = self.h1_to_softmax(h1)
        visible_sample = self.sample_softmax(visible_probs, D)
        return visible_sample    

    def sample_h2(self, h1):
        D = np.ones(h1.shape[0])*self.M
        visible_probs = self.h1_to_softmax(h1, D)
        visible_sample = self.sample_softmax(visible_probs, D)
        return visible_sample    


    def v_and_h2_to_h1(self, v, h2):
        w_vh, w_v, w_h = self.W
        D = v.sum(axis=1)
        energy = (np.outer(w_h , (D + self.M)) + w_vh.T @ (v + h2).T).T  #N x F
        h1 = self.sigmoid(energy)
        return h1

    def v_to_mf_h1(self, v):
        w_vh, w_v, w_h = self.W
        D = v.sum(axis=1)
        energy = np.outer((D + self.M), w_h ) + (v @ w_vh) * np.reshape( (1 + self.M/D) ,(-1,1))  #N x F
        h1 = self.sigmoid(energy)
        return h1
    
    # def visible2hidden(self, v):
    #     return self.v_to_mf_h1(v)

    def visible_to_hiddens_gibbs(self, v):
        '''
        main function to compute the hidden states given visible states
        in the training of the over replicated softmax model.
        Uses mean field approximation to get the expected values of the two hidden layers.
        The third hidden layer is initialized as uniform random.

        v: visible states N x K
        '''

        converge = False
        mu2 = np.random.random(self.K) * self.M #initialize mu2 randomly

        while not converge:
            old_mu2 = mu2
            h2 = mu2 * self.M #self.sample_h2(mu2, np.ones(v.shape[0])*self.M)
            mu1 = self.v_and_h2_to_h1(v, h2)
            mu2 = self.h1_to_softmax(mu1)

            if (old_mu2 - mu2).sum() < self.epsilon:
                converge = True

        return mu1, mu2


    def unif_reject_sample(self, probs):
        h_unif = np.random.rand(*probs.shape)
        h_sample = np.array(h_unif < probs, dtype=int)
        return h_sample

    def sample_hidden(self, v):
        h1_probs = self.visible2hidden_mf(v)
        h1_sample = self.unif_reject_sample(h1_probs)
        return h1_sample
    
##################################### leapfrog trainsition operators


    def gibbs_transition(self, v):
        '''
        makes a gibbs transition on a batch of visible states v
        using the full gibbs sampling for the hidden layers
        '''
        D = v.sum(axis=1)
        hidden_probs = self.visible_to_hiddens_gibbs(v)
        hidden_sample = self.unif_reject_sample(hidden_probs)
        visible_probs = self.h1_to_softmax(hidden_sample)
        visible_sample = np.empty(v.shape)
        for i in range(v.shape[0]):
            visible_sample[i] = self.multinomial_sample(visible_probs[i], D[i])
        return visible_sample


    def gibbs_transition_lowcost(self, v):
        '''
        makes a gibbs transition on a batch of visible states v
        using the mean field approximation for the hidden layers
        '''
        D = v.sum(axis=1)
        hidden_probs = self.v_to_mf_h1(v)
        hidden_sample = self.unif_reject_sample(hidden_probs)
        visible_probs = self.h1_to_softmax(hidden_sample)
        visible_sample = np.empty(v.shape)
        for i in range(v.shape[0]):
            visible_sample[i] = self.multinomial_sample(visible_probs[i], D[i])
        return visible_sample




##################################### interepret topic-words matrix

    def _get_topic_word_matrix(self):
        """
        Return the topic representation of the words
        """
        w_vh, w_v, w_h = self.W
        topic_word_matrix = w_vh.T
        normalized = []
        for words_w in topic_word_matrix:
            minimum = min(words_w)
            words = words_w - minimum
            normalized.append([float(i)/sum(words) for i in words])
        topic_word_matrix = np.array(normalized)
        return topic_word_matrix




######################## gradient descent optimization


    def interaction_penalty(self, vel_vh, w_vh):
        if self.penalty:
            if self.penL1: #L1 penalty
                if self.local_penalty:
                    penal = self.decay*np.sign(w_vh)
                else:
                    penal = self.decay*np.sum(np.abs(w_vh))*np.sign(w_vh)
            else:          #L2 penalty
                if self.local_penalty:
                    penal = self.decay*w_vh
                else:
                    penal = self.decay*np.sum(w_vh)

            vel_vh = vel_vh - penal
        return vel_vh



    def gradient_simple(self, v1, v2, h11, h12 , h21, h22):
        w_vh, w_v, w_h = self.W
        lr = self.lr

        vel_vh = np.dot((v1 + h21).T, h11) - np.dot((v2+h22).T, h12)
        vel_vh = self.interaction_penalty(vel_vh, w_vh)

        vel_v = (v1 + h21).sum(axis=0) - (v2 + h22).sum(axis=0)
        vel_h = h11.sum(axis=0) - h12.sum(axis=0)

        w_vh += vel_vh * lr
        w_v += vel_v * lr
        w_h += vel_h * lr
        
        self.W = w_vh, w_v, w_h


    def gradient_momentum(self, v1, v2, h11, h12 , h21, h22):
        w_vh, w_v, w_h = self.W
        vel_vh, vel_v, vel_h = self.train_cache
        m = self.momentum
        lr = self.lr

        vel_vh = vel_vh * m + (np.dot((v1 + h21).T, h11) - np.dot((v2+h22).T, h12)) * (1-m)
        vel_vh = self.interaction_penalty(vel_vh, w_vh)
        vel_v = vel_v * m + ((v1 + h21).sum(axis=0) - (v2 + h22).sum(axis=0)) * (1-m)
        vel_h = vel_h * m + (h11.sum(axis=0) - h12.sum(axis=0)) * (1-m)

        w_vh += vel_vh * lr
        w_v += vel_v * lr
        w_h += vel_h * lr
        
        self.W = w_vh, w_v, w_h
        self.train_cache = vel_vh, vel_v, vel_h



    def gradient_adagrad(self, v1, v2, h11, h12 , h21, h22):
        w_vh, w_v, w_h = self.W
        vel_vh, vel_v, vel_h = self.train_cache
        m = self.momentum
        lr = self.lr

        vel_vh = np.dot((v1 + h21).T, h11) - np.dot((v2+h22).T, h12)
        vel_vh = self.interaction_penalty(vel_vh, w_vh)
        vel_v = (v1 + h21).sum(axis=0) - (v2 + h22).sum(axis=0)
        vel_h = h11.sum(axis=0) - h12.sum(axis=0)


        w_vh += vel_vh * lr / (np.sqrt(np.sum(vel_vh**2)) + 1e-8)
        w_v += vel_v * lr / (np.sqrt(np.sum(vel_v**2)) + 1e-8)
        w_h += vel_h * lr / (np.sqrt(np.sum(vel_h**2)) + 1e-8)

        self.W = w_vh, w_v, w_h
        self.train_cache = vel_vh, vel_v, vel_h   




    def gradient_rmsprop(self, v1, v2, h11, h12 , h21, h22):
        w_vh, w_v, w_h,  = self.W
        vel_vh, vel_v, vel_h, rms_m2_vh, rms_m2_v, rms_m2_h = self.train_cache
        m = self.momentum
        rms_decay = self.rms_decay
        lr = self.lr

        vel_vh = np.dot((v1 + h21).T, h11) - np.dot((v2+h22).T, h12)
        vel_vh = self.interaction_penalty(vel_vh, w_vh)
        vel_v = (v1 + h21).sum(axis=0) - (v2 + h22).sum(axis=0)
        vel_h = h11.sum(axis=0) - h12.sum(axis=0)

        rms_m2_vh = rms_decay * rms_m2_vh + (1 - rms_decay) * (vel_vh**2)
        w_vh += lr * vel_vh / np.sqrt(rms_m2_vh + 1e-8)
        rms_m2_v = rms_decay * rms_m2_v + (1 - rms_decay) * (vel_v**2)
        w_v += lr * vel_v / np.sqrt(rms_m2_v + 1e-8)
        rms_m2_h = rms_decay * rms_m2_h + (1 - rms_decay) * (vel_h**2)
        w_h += lr * vel_h / np.sqrt(rms_m2_h + 1e-8)

        self.W = w_vh, w_v, w_h
        self.train_cache = vel_vh, vel_v, vel_h, rms_m2_vh, rms_m2_v, rms_m2_h   



    def gradient_adam(self, v1, v2, h11, h12 , h21, h22):
        w_vh, w_v, w_h = self.W
        vel_vh, vel_v, vel_h, adam_m1_vh, adam_m1_v, adam_m1_h, adam_m2_vh, adam_m2_v, adam_m2_h, t = self.train_cache
        decay1 = self.adam_decay1
        decay2 = self.adam_decay2
        lr = self.lr

        vel_vh = np.dot((v1 + h21).T, h11) - np.dot((v2+h22).T, h12)
        vel_vh = self.interaction_penalty(vel_vh, w_vh)
        vel_v = (v1 + h21).sum(axis=0) - (v2 + h22).sum(axis=0)
        vel_h = h11.sum(axis=0) - h12.sum(axis=0)

        # Increment t first (should start from 1, not 0)
        t += 1
        
        # Compute bias correction terms
        bias_correction1 = 1 - decay1**t
        bias_correction2 = 1 - decay2**t

        # Update for w_vh
        adam_m1_vh = decay1 * adam_m1_vh + (1 - decay1) * vel_vh
        adam_m2_vh = decay2 * adam_m2_vh + (1 - decay2) * (vel_vh**2)
        adam_m1_vh_hat = adam_m1_vh / bias_correction1
        adam_m2_vh_hat = adam_m2_vh / bias_correction2
        w_vh += lr * adam_m1_vh_hat / (np.sqrt(adam_m2_vh_hat) + 1e-8)

        # Update for w_v
        adam_m1_v = decay1 * adam_m1_v + (1 - decay1) * vel_v
        adam_m2_v = decay2 * adam_m2_v + (1 - decay2) * (vel_v**2)
        adam_m1_v_hat = adam_m1_v / bias_correction1
        adam_m2_v_hat = adam_m2_v / bias_correction2
        w_v += lr * adam_m1_v_hat / (np.sqrt(adam_m2_v_hat) + 1e-8)

        # Update for w_h
        adam_m1_h = decay1 * adam_m1_h + (1 - decay1) * vel_h
        adam_m2_h = decay2 * adam_m2_h + (1 - decay2) * (vel_h**2)
        adam_m1_h_hat = adam_m1_h / bias_correction1
        adam_m2_h_hat = adam_m2_h / bias_correction2
        w_h += lr * adam_m1_h_hat / (np.sqrt(adam_m2_h_hat) + 1e-8)

        self.W = w_vh, w_v, w_h
        self.train_cache = vel_vh, vel_v, vel_h, adam_m1_vh, adam_m1_v, adam_m1_h, adam_m2_vh, adam_m2_v, adam_m2_h, t



####################### contrastive divergence steps

##### cd steps for training

    def kcd_step(self, v, K):
        v = self.gibbs_transition(v)
        h1, mu2 = self.visible_to_hiddens_gibbs(v)
        h2 = mu2 * self.M #self.sample_h2(mu2, np.ones(v.shape[0])*self.M)

        D = v.sum(axis=1)
        for k in range(K):
            v_model = self.sample_visible(h1, D)
            h1_model, mu2_model = self.visible_to_hiddens_gibbs(v_model)
            
        h2_model = mu2_model * self.M
        self.gradient_step(v, v_model, h1, h1_model, h2,  h2_model)


    def pcd_step(self, v0, pv0):
        D = v0.sum(axis=1)
        h1, h2 = self.visible_to_hiddens_gibbs(v0)
        pv1 = self.gibbs_transition(pv0)
        ph1, ph2 = self.visible_to_hiddens_gibbs(pv1)
        h2 = h2 * self.M
        ph2 = ph2 * self.M
        self.gradient_step(v0,pv1,h1,ph1, h2, ph2)
        return pv1


    def mfcd_step(self, v0):
        D = v0.sum(axis=1)
        h0, mu0 = self.visible_to_hiddens_gibbs(v0)
        v1 = self.h1_to_softmax(h0) * D.reshape(-1, 1)
        h1, mu1 = self.visible_to_hiddens_gibbs(v1)
        mu0 = mu0 * self.M
        mu1 = mu1 * self.M
        self.gradient_step(v0,v1,h0,h1, mu0, mu1)





##### cd steps for pre-training


    def pretrain_kcd_step(self, v, K):
        h1 = self.v_to_mf_h1(v)
        D = v.sum(axis=1)
        h2 = v * self.M/ D.reshape(-1, 1) #self.sample_h2(mu2, np.ones(v.shape[0])*self.M)

        for k in range(K):
            v_model = self.sample_visible(h1, D)
            h1_model = self.v_to_mf_h1(v_model)
            
        mu2_model = self.h1_to_softmax(h1_model)
        h2_model = mu2_model * self.M
        self.gradient_step(v, v_model, h1, h1_model, h2,  h2_model)



    def pretrain_mfcd_step(self, v0):
        D = v0.sum(axis=1)
        h0 = self.v_to_mf_h1(v0)
        v1 = self.h1_to_softmax(h0) * D.reshape(-1, 1)
        h1 = self.v_to_mf_h1(v1)
        self.gradient_step(v0,v1,h0,h1, v0*self.M/D.reshape(-1, 1), v1*self.M/D.reshape(-1, 1))



    def pretrain_pcd_step(self, v0, pv0):
        D = v0.sum(axis=1)
        h0 = self.v_to_mf_h1(v0)
        pv1 = self.gibbs_transition_lowcost(pv0)
        ph1 = self.v_to_mf_h1(pv1)
        self.gradient_step(v0,pv1,h0,ph1, v0*self.M/D.reshape(-1, 1), pv1*self.M/D.reshape(-1, 1))
        return pv1




############################### main train function



    def train(self, dtm, num_topics, epochs, M,  pretrain_epochs=1,
             btsz=100, lr=0.01, momentum=0.1, initw=None, 
            softstart = 0.001, epsilon=0.01, K=1,
            decay=0, penalty_L1=False, penalty_local=False, 
            val_dtm=None, monitor_time=True, monitor_ppl=False,  increase_speed = 0,
            train_optimizer='sgd', cd_type='mfcd', logdtm=False,
            rms_decay=0.9,adam_decay1=0.9, adam_decay2=0.999,
            epochs_per_monitor=1, random_state=None):

        hidden = num_topics
        self.F = hidden
        self.hidden = hidden
        self.K = K
        N, dictsize = dtm.shape
        self.M = M
        self.momentum = momentum
        self.lr = lr
        batches = int(np.floor(N/btsz))
        self.epsilon = epsilon
        self.decay = decay
        self.penalty = decay > 0
        self.penL1 = penalty_L1
        self.local_penalty = penalty_local

        self.train_optimizer = train_optimizer
        self.adam_decay1 = adam_decay1
        self.adam_decay2 = adam_decay2
        self.rms_decay = rms_decay


        self.persist = (cd_type=='persistent') #persistent_cd
        self.mean_field = (cd_type=='mfcd') #mean_field_cd
        self.gradual = (cd_type=='gradcd') #increase_cd


        doval = (val_dtm is not None)

        if random_state is not None:
            np.random.seed(random_state)



        if monitor_time:
            self.train_time = np.empty(epochs)

        if monitor_ppl:
            monit_epochs = np.arange(stop = epochs, step = epochs_per_monitor)
            next_monitor = 0
            self.train_loglik = np.empty(len(monit_epochs))
            self.train_ppl = np.empty(len(monit_epochs))
            if doval:
                self.val_loglik = np.empty(len(monit_epochs))
                self.val_ppl = np.empty(len(monit_epochs))



        ## initialize k
        if self.gradual:
            Kvec = self.gradual_kcd(T=epochs, K=self.K, g=increase_speed)
        else:
            Kvec = np.ones(epochs)*self.K
        Kvec = Kvec.astype(int)

        # Initialize persistent chain - one chain for each document in the dataset
        # Each persistent visible should have the same document length as corresponding data
        if self.persist:
            persistent_v = np.zeros((N, dictsize))  # Full dataset size
            persistent_D = dtm.sum(axis=1)  # Document lengths from original data
            
            # Initialize each document with uniform multinomial of its actual length
            for i in range(N):
                if persistent_D[i] > 0:  # Avoid empty documents
                    persistent_v[i] = np.random.multinomial(persistent_D[i], np.ones(dictsize)/dictsize)
    
        obs_ids = np.arange(N)



        if initw is not None:
            self.W = initw

        if self.W is None:
            w_vh = softstart * np.random.randn(dictsize, hidden)
            w_v = softstart * np.random.randn(dictsize)
            w_h = softstart * np.random.randn(hidden)
        else:
            print('train already available weights')
            w_vh, w_v, w_h = self.W

        vel_vh = np.zeros((dictsize, hidden))
        vel_v = np.zeros((dictsize))
        vel_h = np.zeros((hidden))

        self.W = w_vh, w_v, w_h
        self.velocities = vel_vh, vel_v, vel_h

        obs_ids = np.arange(N)

        if self.train_optimizer == 'sgd':
            self.gradient_step = self.gradient_simple
        else:
            if self.train_optimizer == 'momentum':
                self.gradient_step = self.gradient_momentum
                self.train_cache = vel_vh, vel_v, vel_h
            else:
                if self.train_optimizer == 'adagrad':
                    self.gradient_step = self.gradient_adagrad
                    self.train_cache = vel_vh, vel_v, vel_h
                else:
                    if self.train_optimizer == 'rmsprop':
                        self.gradient_step = self.gradient_rmsprop
                        rms_m2_vh = np.zeros((dictsize, hidden))
                        rms_m2_v = np.zeros((dictsize))
                        rms_m2_h = np.zeros((hidden))
                        self.rms_decay = 0.9
                        self.train_cache = vel_vh, vel_v, vel_h, rms_m2_vh, rms_m2_v, rms_m2_h
                    else:
                        if self.train_optimizer == 'adam':
                            self.gradient_step = self.gradient_adam
                            adam_m1_vh = np.zeros((dictsize, hidden))
                            adam_m1_v = np.zeros((dictsize))
                            adam_m1_h = np.zeros((hidden))
                            adam_m2_vh = np.zeros((dictsize, hidden))
                            adam_m2_v = np.zeros((dictsize))
                            adam_m2_h = np.zeros((hidden))
                            t = 1
                            self.adam_decay1 = 0.9
                            self.adam_decay2 = 0.999
                            self.train_cache = vel_vh, vel_v, vel_h, adam_m1_vh, adam_m1_v, adam_m1_h, adam_m2_vh, adam_m2_v, adam_m2_h, t
                        else:
                            self.gradient_step = self.gradient_simple

        if logdtm:
            dtm = np.log(1 + dtm)
            if doval:
                val_dtm = np.log(1 + val_dtm)


        ## MAIN TRAIN LOOP
        print("Training OverRS model...")



        ##loop
        for t in tqdm(range(epochs)):
            
            if monitor_time:
                current_time = time.time()

            start_id = 0
            np.random.shuffle(obs_ids) # apply sgd
            dtm = dtm[obs_ids,:]
            if self.persist:
                persistent_v = persistent_v[obs_ids,:]

            if t < pretrain_epochs:
                for b in range(batches):
                    v = dtm[start_id : start_id + btsz , :]
                    #self.pretrain_kcd_step(v)

                    if self.mean_field:
                        self.pretrain_mfcd_step(v)
                    else:
                        if self.persist:
                            vp = persistent_v[start_id : start_id + btsz , :]
                            persistent_v[start_id : start_id + btsz , :] = self.pretrain_pcd_step(v, vp)
                        else:
                            self.pretrain_kcd_step(v, Kvec[t])

                        start_id += btsz
            else:
                for b in range(batches):
                    v = dtm[start_id : start_id + btsz , :]
                    #self.kcd_step(v)
                    if self.mean_field:
                        self.mfcd_step(v)
                    else:
                        if self.persist:
                            vp = persistent_v[start_id : start_id + btsz , :]
                            persistent_v[start_id : start_id + btsz , :] = self.pcd_step(v, vp)
                        else:
                            self.kcd_step(v, Kvec[t])

                        start_id += btsz
            

            if monitor_time:
                elapsed_time = time.time() - current_time
                self.train_time[t] = elapsed_time

            if monitor_ppl:
                if t == monit_epochs[next_monitor]:
                    next_monitor += 1
                    next_monitor = t + epochs_per_monitor

                    self.train_loglik[t] = np.mean(self.neg_free_energy(dtm))
                    self.train_ppl[t] = self.log_ppl_upbo(dtm)

                    if doval:
                        self.val_loglik[t] = np.mean(self.neg_free_energy(val_dtm))
                        self.val_ppl[t] = self.log_ppl_upbo(val_dtm)


    def log_ppl_upbo(self, dtm):
        """
        return the log perplepxity upper bound 
        given a document term matrix
        """
        mfh = self.v_to_mf_h1(dtm)
        vprob = self.h1_to_softmax(mfh)
        lpub = np.exp(-np.nansum(np.log(vprob)*dtm)/np.sum(dtm))
        return lpub
    

    def topic_words(self, topk, id2word=None):
        w_vh, w_v, w_h = self.W
        T = self.hidden
        if id2word==None:
            id2word = self.id2word
        words = np.array([k for k in id2word.token2id.keys()])

        toplist = []
        for t in range(T):
            topw = w_vh[: , t]
            bestwords = words[np.argsort(topw)[::-1]][0:topk]
            toplist.append(bestwords)

        return toplist


