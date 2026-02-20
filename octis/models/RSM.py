from octis.models.model import AbstractModel
from octis.models.RS_class import Replicated_Softmax
import numpy as np
from tqdm import tqdm
import gensim.corpora as corpora
import octis.configuration.citations as citations
import octis.configuration.defaults as defaults
import time
import warnings


################## RSM octis class


class RSM(AbstractModel):
    id2word = None
    id_corpus = None
    use_partitions = True
    update_with_test = False

    def __init__(
        self,
        num_topics=50,
        epochs=5,
        btsz=100,
        lr=0.01,
        momentum=0.9,
        K=1,
        softstart=0.001,
        decay=0,
        penalty_L1=False,
        penalty_local=False,
        monitor_ppl=False,
        monitor_time=False,
        monitor_loglik=False,
        increase_speed=1,
        rms_decay=0.9,
        adam_decay1=0.9,
        adam_decay2=0.999,
        cd_type="mfcd",
        train_optimizer="sgd",
        verbose=False,
        logdtm=False,
        random_state=None,
    ):
        """
        Parameters
        ----------
        num_topics : number of topics
        epochs : number of training epochs
        btsz : batch size
        lr : learning rate
        momentum : momentum of momentum optimizer
        (applied only if train_optimizer='momentum')
        rms_decay : decay rate for RMSProp optimizer
        (applied only if train_optimizer='rmsprop')
        adam_decay1 : first decay rate for Adam optimizer
        (applied only if train_optimizer='adam')
        adam_decay2 : second decay rate for Adam optimizer
        (applied only if train_optimizer='adam')
        K : number of Gibbs sampling steps when using KCD
        decay : penalization coefficient, default 0 (no penalization)
        penalty_L1 : if True uses L1 penalization, else L2 penalization
        penalty_local : if True uses local penalization,
        else global penalization
        softstart : initialization scale for weights
        (randomly drawn from N(0,1)*softstart)
        logdtm : if True each cell of the dtm is transformed as log(1+cell),
        otherwise the raw counts are used
        monitor : if True prints training information during training

        cd_type : type of contrastive divergence to use,
          'kcd', 'pcd', 'mfcd' (default) or 'gradcd' :
                    'kcd' stands for k-step contrastive divergence
                    'pcd' stands for persistent contrastive divergence
                    'mfcd' stands for mean-field contrastive divergence
                    'gradcd' stands for gradual k-step contrastive divergence,
                    where k increases over epochs latter when increase_speed is higher
        train_optimizer : training optimizer to use :
                    'full' for full batch training,
                    'sgd' for simple stochastic gradient descent,
                    'minibatch' for mini-batch training,
                    'momentum' for mini-batch with momentum,
                    'rmsprop' for RMSProp optimizer,
                    'adam' for Adam optimizer,
                    'adagrad' for Adagrad optimizer


        Example usage
        --------------------

        from octis.dataset.dataset import Dataset
        from octis.models.RSM import RSM

        dataset_20ng = Dataset()
        dataset_20ng.fetch_dataset("20NewsGroup")

        rsm = RSM(num_topics=20, epochs=500, btsz=20, lr=0.0001, cd_type='mfcd', train_optimizer='rmsprop')
        output_rsm = rsm.train(dataset_20ng)
        """
        super().__init__()
        self.hyperparameters = dict()
        self.hyperparameters["num_topics"] = num_topics
        self.hyperparameters["btsz"] = btsz
        self.hyperparameters["lr"] = lr
        self.hyperparameters["momentum"] = momentum
        self.hyperparameters["K"] = K
        self.hyperparameters["softstart"] = softstart
        self.hyperparameters["epochs"] = epochs
        self.hyperparameters["increase_speed"] = increase_speed
        self.hyperparameters["monitor_time"] = monitor_time
        self.hyperparameters["monitor_ppl"] = monitor_ppl
        self.hyperparameters["monitor_loglik"] = monitor_loglik
        self.hyperparameters["penalty_L1"] = penalty_L1
        self.hyperparameters["penalty_local"] = penalty_local
        self.hyperparameters["decay"] = decay
        self.hyperparameters["random_state"] = random_state
        self.hyperparameters["cd_type"] = cd_type
        self.hyperparameters["logdtm"] = logdtm
        self.hyperparameters["val_dtm"] = None
        self.hyperparameters["train_optimizer"] = train_optimizer
        self.hyperparameters["rms_decay"] = rms_decay
        self.hyperparameters["adam_decay1"] = adam_decay1
        self.hyperparameters["adam_decay2"] = adam_decay2
        self.hyperparameters["verbose"] = verbose

    def info(self):
        """
        Returns model informations
        """
        return {
            "citation": citations.models_RSM,
            "name": "RSM, Replicated Softmax Model",
        }

    def hyperparameters_info(self):
        """
        Returns hyperparameters informations
        """
        return defaults.RSM_hyperparameters_info

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

        self.initialize_model_structure(hyperparams=hyperparams, dataset=dataset)

        self.trained_model.train(**self.hyperparameters)

        return self.get_model_output(top_words)

    def get_model_output(self, top_words=10):
        result = {}

        result["topic-word-matrix"] = self.trained_model._get_topic_word_matrix()

        if top_words > 0:
            result["topics"] = self.trained_model._get_topics(top_words)

        result["topic-document-matrix"] = self.trained_model._get_topic_doc(
            self.train_dtm
        )

        if self.use_partitions:
            result["test-topic-document-matrix"] = self.trained_model._get_topic_doc(
                self.test_dtm
            )
        else:
            result["test-topic-document-matrix"] = result["topic-document-matrix"]

        return result

    def initialize_model_structure(self, hyperparams, dataset):
        if hyperparams is None:
            hyperparams = {}

        if self.use_partitions:
            train_corpus, test_corpus = dataset.get_partitioned_corpus(
                use_validation=False
            )
        else:
            train_corpus = dataset.get_corpus()

        if self.id2word is None:
            self.id2word = self.get_vocab(dataset.get_corpus())

        if self.use_partitions:
            if self.hyperparameters["verbose"]:
                print("Building train DTM...")
            self.train_dtm = self.build_dtm(train_corpus, self.id2word)
            if self.hyperparameters["verbose"]:
                print("Building test DTM...")
            self.test_dtm = self.build_dtm(test_corpus, self.id2word)
            hyperparams["dtm"] = self.train_dtm
            hyperparams["val_dtm"] = self.test_dtm
        else:
            if self.hyperparameters["verbose"]:
                print("Building DTM...")
            self.train_dtm = self.build_dtm(train_corpus, self.id2word)
            hyperparams["dtm"] = self.train_dtm
            hyperparams["val_dtm"] = None

        if "num_topics" not in hyperparams:
            hyperparams["num_topics"] = self.hyperparameters["num_topics"]

        self.hyperparameters.update(hyperparams)

        self.trained_model = self.RSM_model()
        self.trained_model.id2word = self.id2word

    ############### preprocessing functions

    def get_vocab(self, tokenized_corpus):
        id2word = corpora.Dictionary(tokenized_corpus)
        return id2word

    def build_dtm(self, tokenized_corpus, id2word=None):
        """
        converts a tokenized corpus to a DOcument Term Matrix. id2word is a gensim dictionary.
        """
        if id2word is None:
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
                DTM[i, id] = count

        if self.hyperparameters["logdtm"]:
            DTM = np.log(1 + DTM)
        return DTM

    ##############################################################  RSM original class

    class RSM_model(Replicated_Softmax):
        def __init__(self):
            super().__init__()

        ############################## energy and probability

        def neg_energy(self, v, h):
            w_vh, w_v, w_h = self.W
            D = v.sum(axis=1)
            t1 = v @ w_v
            t2 = D * (h @ w_h)
            t3 = (v @ w_vh @ h.T).sum(axis=1)
            en = t1 + t2 + t3
            return en

        def visible2hidden_vec(self, v):
            w_vh, w_v, w_h = self.W
            D = v.sum()
            energy = D * w_h + np.dot(v, w_vh)
            return self.sigmoid(energy)

        def visible2hidden(self, v):
            w_vh, w_v, w_h = self.W
            D = np.tile(v.sum(axis=1), (w_h.shape[0], 1)).T
            energy = D * w_h + np.dot(v, w_vh)
            return self.sigmoid(energy)

        def hidden2visible_vec(self, h):
            w_vh, w_v, w_h = self.W
            energy = w_v + np.dot(w_vh, h)
            return self.softmax_vec(energy)

        def hidden2visible(self, h):
            w_vh, w_v, w_h = self.W
            energy = np.tile(w_v, (h.shape[0], 1)).T + np.dot(w_vh, h.T)
            return self.softmax(energy.T)

        ##################################### leapfrog trainsition operators

        def gibbs_transition(self, v):
            D = v.sum(axis=1)
            hidden_probs = self.visible2hidden(v)
            hidden_sample = self.unif_reject_sample(hidden_probs)
            visible_probs = self.hidden2visible(hidden_sample)
            visible_sample = np.empty(v.shape)
            for i in range(v.shape[0]):
                visible_sample[i] = self.multinomial_sample(visible_probs[i], D[i])
            return visible_sample

        def MH_transition(self, state, logpdf):
            new = self.gibbs_transition(state)
            old_logpdf = logpdf(state)
            new_logpdf = logpdf(new)

            accept_ratio = min(1, np.exp(new_logpdf - old_logpdf))

            # Accept or reject
            if np.random.random() < accept_ratio:
                return new
            else:
                return state

        #### leapfrog for single document vectors (useful for ais estimates of perplexity)

        def gibbs_transition_vec(self, v):
            D = v.sum()
            hidden_probs = self.visible2hidden_vec(v)
            hidden_sample = self.unif_reject_sample(hidden_probs)
            visible_probs = self.hidden2visible_vec(hidden_sample)
            visible_sample = self.multinomial_sample(visible_probs, D)
            return visible_sample

        def MH_transition_vec(self, state, logpdf):
            new = self.gibbs_transition_vec(state)
            old_logpdf = logpdf(state)
            new_logpdf = logpdf(new)

            accept_ratio = min(1, np.exp(new_logpdf - old_logpdf))

            # Accept or reject
            if np.random.random() < accept_ratio:
                return new
            else:
                return state

        ################################## gradient descent optimization

        def gradient_simple(self, v1, v2, h1, h2):
            w_vh, w_v, w_h = self.W
            lr = self.lr

            vel_vh = np.dot(v1.T, h1) - np.dot(v2.T, h2)
            vel_vh = self.interaction_penalty(vel_vh, w_vh)
            vel_v = v1.sum(axis=0) - v2.sum(axis=0)
            vel_h = h1.sum(axis=0) - h2.sum(axis=0)

            w_vh += vel_vh * lr
            w_v += vel_v * lr
            w_h += vel_h * lr

            if any(
                (np.any(np.isnan(w_vh)), np.any(np.isnan(w_v)), np.any(np.isnan(w_h)))
            ):
                self.stop = True
                warnings.warn("NaN values founded in weights: stopping training")
            else:
                self.W = w_vh, w_v, w_h

        def gradient_momentum(self, v1, v2, h1, h2):
            w_vh, w_v, w_h = self.W
            vel_vh, vel_v, vel_h = self.train_cache
            m = self.momentum
            lr = self.lr

            vel_vh = vel_vh * m + (np.dot(v1.T, h1) - np.dot(v2.T, h2)) * (1 - m)
            vel_vh = self.interaction_penalty(vel_vh, w_vh)
            vel_v = vel_v * m + (v1.sum(axis=0) - v2.sum(axis=0)) * (1 - m)
            vel_h = vel_h * m + (h1.sum(axis=0) - h2.sum(axis=0)) * (1 - m)

            w_vh += vel_vh * lr
            w_v += vel_v * lr
            w_h += vel_h * lr

            if any(
                (np.any(np.isnan(w_vh)), np.any(np.isnan(w_v)), np.any(np.isnan(w_h)))
            ):
                self.stop = True
                warnings.warn("NaN values founded in weights: stopping training")
            else:
                self.W = w_vh, w_v, w_h

            self.train_cache = vel_vh, vel_v, vel_h

        def gradient_adagrad(self, v1, v2, h1, h2):
            w_vh, w_v, w_h = self.W
            vel_vh, vel_v, vel_h = self.train_cache
            lr = self.lr

            vel_vh = np.dot(v1.T, h1) - np.dot(v2.T, h2)
            vel_vh = self.interaction_penalty(vel_vh, w_vh)
            vel_v = v1.sum(axis=0) - v2.sum(axis=0)
            vel_h = h1.sum(axis=0) - h2.sum(axis=0)

            w_vh += vel_vh * lr / (np.sqrt(np.sum(vel_vh**2)) + 1e-8)
            w_v += vel_v * lr / (np.sqrt(np.sum(vel_v**2)) + 1e-8)
            w_h += vel_h * lr / (np.sqrt(np.sum(vel_h**2)) + 1e-8)

            if any(
                (np.any(np.isnan(w_vh)), np.any(np.isnan(w_v)), np.any(np.isnan(w_h)))
            ):
                self.stop = True
                warnings.warn("NaN values founded in weights: stopping training")
            else:
                self.W = w_vh, w_v, w_h

            self.train_cache = vel_vh, vel_v, vel_h

        def gradient_rmsprop(self, v1, v2, h1, h2):
            (
                w_vh,
                w_v,
                w_h,
            ) = self.W
            vel_vh, vel_v, vel_h, rms_m2_vh, rms_m2_v, rms_m2_h = self.train_cache
            rms_decay = self.rms_decay
            lr = self.lr

            vel_vh = np.dot(v1.T, h1) - np.dot(v2.T, h2)
            vel_vh = self.interaction_penalty(vel_vh, w_vh)
            vel_v = v1.sum(axis=0) - v2.sum(axis=0)
            vel_h = h1.sum(axis=0) - h2.sum(axis=0)

            rms_m2_vh = rms_decay * rms_m2_vh + (1 - rms_decay) * (vel_vh**2)
            w_vh += lr * vel_vh / np.sqrt(rms_m2_vh + 1e-8)
            rms_m2_v = rms_decay * rms_m2_v + (1 - rms_decay) * (vel_v**2)
            w_v += lr * vel_v / np.sqrt(rms_m2_v + 1e-8)
            rms_m2_h = rms_decay * rms_m2_h + (1 - rms_decay) * (vel_h**2)
            w_h += lr * vel_h / np.sqrt(rms_m2_h + 1e-8)

            if any(
                (np.any(np.isnan(w_vh)), np.any(np.isnan(w_v)), np.any(np.isnan(w_h)))
            ):
                self.stop = True
                warnings.warn("NaN values founded in weights: stopping training")
            else:
                self.W = w_vh, w_v, w_h

            self.train_cache = vel_vh, vel_v, vel_h, rms_m2_vh, rms_m2_v, rms_m2_h

        def gradient_adam(self, v1, v2, h1, h2):
            w_vh, w_v, w_h = self.W
            (
                vel_vh,
                vel_v,
                vel_h,
                adam_m1_vh,
                adam_m1_v,
                adam_m1_h,
                adam_m2_vh,
                adam_m2_v,
                adam_m2_h,
                t,
            ) = self.train_cache
            decay1 = self.adam_decay1
            decay2 = self.adam_decay2
            lr = self.lr

            vel_vh = np.dot(v1.T, h1) - np.dot(v2.T, h2)
            vel_vh = self.interaction_penalty(vel_vh, w_vh)
            vel_v = v1.sum(axis=0) - v2.sum(axis=0)
            vel_h = h1.sum(axis=0) - h2.sum(axis=0)

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

            if any(
                (np.any(np.isnan(w_vh)), np.any(np.isnan(w_v)), np.any(np.isnan(w_h)))
            ):
                self.stop = True
                warnings.warn("NaN values founded in weights: stopping training")
            else:
                self.W = w_vh, w_v, w_h

            self.train_cache = (
                vel_vh,
                vel_v,
                vel_h,
                adam_m1_vh,
                adam_m1_v,
                adam_m1_h,
                adam_m2_vh,
                adam_m2_v,
                adam_m2_h,
                t,
            )

        ########################################## contrastive divergence steps

        def kcd_step(self, ids):
            v0 = self.dtm[ids, :]
            h0 = self.visible2hidden(v0)
            v1 = v0
            for k in range(self.tK):
                v1 = self.gibbs_transition(v1)
            h1 = self.visible2hidden(v1)

            if not self.mean_h:  # converting probabilities to binaries
                h0 = self.unif_reject_sample(h0)
                h1 = self.unif_reject_sample(h1)

            self.gradient_step(v0, v1, h0, h1)

        def mfcd_step(self, ids):
            v0 = self.dtm[ids, :]
            D = v0.sum(axis=1)
            h0 = self.visible2hidden(v0)
            v1 = self.hidden2visible(h0) * D.reshape(-1, 1)
            h1 = self.visible2hidden(v1)

            self.gradient_step(v0, v1, h0, h1)

        def gradkcd_step(self, ids):
            self.tK = self.Kvec[self.t]
            if self.tK == 0:
                self.mfcd_step(ids)
            else:
                self.kcd_step(ids)

        def pcd_step(self, ids):
            v0 = self.dtm[ids, :]
            pv0 = self.persistent_v[ids, :]
            h0 = self.visible2hidden(v0)
            pv1 = self.gibbs_transition(pv0)
            ph1 = self.visible2hidden(pv1)
            self.persistent_v[ids, :] = pv1

            self.gradient_step(v0, pv1, h0, ph1)

        def gradual_k(self, T, K, g=0):
            t = np.arange(1, T + 1)
            k = np.floor((K + 1) * ((t / (T + 1)) ** (1 + g))).astype(int)
            return k

        ########################################## main training function

        def train(
            self,
            dtm,
            num_topics=5,
            epochs=3,
            btsz=100,
            lr=0.01,
            momentum=0.5,
            K=1,
            decay=0,
            penalty_L1=False,
            penalty_local=False,
            monitor_time=False,
            monitor_ppl=False,
            monitor_loglik=False,
            train_optimizer="sgd",
            cd_type="mfcd",
            logdtm=False,
            rms_decay=0.9,
            adam_decay1=0.9,
            adam_decay2=0.999,
            increase_speed=0,
            softstart=0.001,
            winit=None,
            val_dtm=None,
            random_state=None,
            verbose=False,
        ):
            ## init global variables
            if random_state is not None:
                np.random.seed(random_state)

            doval = val_dtm is not None

            # init structure of the model
            self.set_structure_from_dtm(
                winit=winit,
                softstart=softstart,
                epochs=epochs,
                num_topics=num_topics,
                dtm=dtm,
                val_dtm=val_dtm,
                monitor_ppl=monitor_ppl,
                monitor_loglik=monitor_loglik,
                monitor_time=monitor_time,
                logdtm=logdtm,
            )

            ##init training hyperparams
            self.set_train_hyper(
                epochs=epochs,
                btsz=btsz,
                lr=lr,
                momentum=momentum,
                K=K,
                decay=decay,
                penalty_L1=penalty_L1,
                penalty_local=penalty_local,
                train_optimizer=train_optimizer,
                cd_type=cd_type,
                rms_decay=rms_decay,
                adam_decay1=adam_decay1,
                adam_decay2=adam_decay2,
                increase_speed=increase_speed,
            )

            ## MAIN TRAIN LOOP
            print("Training RS model...")
            for t in tqdm(range(epochs)):
                if monitor_time:
                    current_time = time.time()

                if self.stop:
                    print("training stopped early")
                    break
                else:
                    self.train_epoch()

                if monitor_time:
                    elapsed_time = time.time() - current_time
                    self.train_time[t] = elapsed_time

                if monitor_ppl:
                    self.train_ppl[t] = self.log_ppl_approx(dtm)

                    if doval:
                        self.val_ppl[t] = self.log_ppl_approx(val_dtm)

                if monitor_loglik:
                    self.train_loglik[t] = np.mean(self.neg_free_energy(dtm))

                    if doval:
                        self.val_loglik[t] = np.mean(self.neg_free_energy(val_dtm))

        def train_epoch(self):
            """one epoch of training, with sgd and mini-batches"""
            start_id = 0

            # if self.sgd:
            np.random.shuffle(self.obs_ids)  # apply sgd
            self.dtm = self.dtm[self.obs_ids, :]
            if self.persist:
                self.persistent_v = self.persistent_v[self.obs_ids, :]

            for b in range(self.batches):
                ids = np.arange(start_id, start_id + self.btsz)
                self.cd_learning_step(ids)
                start_id += self.btsz

            self.t += 1

        def set_train_hyper(
            self,
            epochs=3,
            btsz=100,
            lr=0.01,
            momentum=0.9,
            K=1,
            decay=0,
            penalty_L1=False,
            penalty_local=False,
            train_optimizer="sgd",
            cd_type="mfcd",
            rms_decay=0.9,
            adam_decay1=0.9,
            adam_decay2=0.999,
            increase_speed=0,
        ):
            N, dictsize = self.dtm.shape
            num_topics = self.hidden

            self.stop = False
            self.momentum = momentum
            self.lr = lr
            self.decay = decay
            self.penalty = decay > 0
            self.penL1 = penalty_L1
            self.local_penalty = penalty_local

            self.train_optimizer = train_optimizer
            self.adam_decay1 = adam_decay1
            self.adam_decay2 = adam_decay2
            self.rms_decay = rms_decay

            self.persist = cd_type == "pcd"  # persistent_cd
            self.mean_field = cd_type == "mfcd"  # mean_field_cd
            self.gradual = cd_type == "gradcd"  # increase_cd

            self.t = 0  # current epoch
            self.K = K
            self.tK = K  # current k
            self.mean_h = True  # whether to use mean hidden activations or sample them

            self.btsz = btsz
            self.batches = int(np.floor(N / btsz))
            # self.sgd = (train_optimizer!='full')
            # self.bt_correct = (btsz**2)/N    #a bayesian would correct decay for batch size. I'm not a bayesian

            ## initialize k
            if self.gradual:
                Kvec = self.gradual_k(T=epochs, K=self.K, g=increase_speed)
            else:
                Kvec = np.ones(epochs) * self.K
            self.Kvec = Kvec.astype(int)

            # Initialize persistent chain - one chain for each document in the dataset
            # Each persistent visible should have the same document length as corresponding data
            if self.persist:
                self.persistent_v = np.zeros((N, dictsize))  # Full dataset size
                persistent_D = self.dtm.sum(
                    axis=1
                )  # Document lengths from original data

                # Initialize each document with uniform multinomial of its actual length
                for i in range(N):
                    if persistent_D[i] > 0:  # Avoid empty documents
                        self.persistent_v[i] = np.random.multinomial(
                            persistent_D[i], np.ones(dictsize) / dictsize
                        )

            # Initialize weights gradients
            vel_vh = np.zeros((dictsize, num_topics))
            vel_v = np.zeros((dictsize))
            vel_h = np.zeros((num_topics))

            if self.train_optimizer == "sgd":
                self.gradient_step = self.gradient_simple
            else:
                if self.train_optimizer == "momentum":
                    self.gradient_step = self.gradient_momentum
                    self.train_cache = vel_vh, vel_v, vel_h
                else:
                    if self.train_optimizer == "adagrad":
                        self.gradient_step = self.gradient_adagrad
                        self.train_cache = vel_vh, vel_v, vel_h
                    else:
                        if self.train_optimizer == "rmsprop":
                            self.gradient_step = self.gradient_rmsprop
                            rms_m2_vh = np.zeros((dictsize, num_topics))
                            rms_m2_v = np.zeros((dictsize))
                            rms_m2_h = np.zeros((num_topics))
                            self.rms_decay = 0.9
                            self.train_cache = (
                                vel_vh,
                                vel_v,
                                vel_h,
                                rms_m2_vh,
                                rms_m2_v,
                                rms_m2_h,
                            )
                        else:
                            if self.train_optimizer == "adam":
                                self.gradient_step = self.gradient_adam
                                adam_m1_vh = np.zeros((dictsize, num_topics))
                                adam_m1_v = np.zeros((dictsize))
                                adam_m1_h = np.zeros((num_topics))
                                adam_m2_vh = np.zeros((dictsize, num_topics))
                                adam_m2_v = np.zeros((dictsize))
                                adam_m2_h = np.zeros((num_topics))
                                t = 1
                                self.adam_decay1 = 0.9
                                self.adam_decay2 = 0.999
                                self.train_cache = (
                                    vel_vh,
                                    vel_v,
                                    vel_h,
                                    adam_m1_vh,
                                    adam_m1_v,
                                    adam_m1_h,
                                    adam_m2_vh,
                                    adam_m2_v,
                                    adam_m2_h,
                                    t,
                                )
                            else:
                                self.gradient_step = self.gradient_simple

            if self.mean_field:
                self.cd_learning_step = self.mfcd_step  # input is v0
            else:
                if self.persist:
                    self.cd_learning_step = (
                        self.pcd_step
                    )  # input is v0, persistent_v, output is new persistent_v
                else:
                    if cd_type == "kcd":
                        self.cd_learning_step = self.kcd_step  # input is v0, K fixed
                    else:  # gradual kcd
                        if self.gradual:
                            self.cd_learning_step = (
                                self.gradkcd_step
                            )  # input is v0, change K each epoch
                        else:
                            self.cd_learning_step = (
                                self.kcd_step
                            )  # input is v0, K fixed

        ############ perplexity and probability

        def log_ppl_approx(self, dtm):
            """
            return the log perplepxity upper bound
            given a document term matrix
            """
            mfh = self.visible2hidden(dtm)
            vprob = self.hidden2visible(mfh)
            vprob = np.clip(vprob, 1e-12, None)
            sum_dtm = np.sum(dtm)
            assert sum_dtm > 0, "the sum of the dtm s entries has to be positive"
            lpub = -np.nansum(np.log(vprob) * dtm) / sum_dtm
            return lpub

        def ppl_approx(self, testmatrix):
            """
            return the perplepxity upper bound
            given a document term matrix
            """

            w_vh, w_v, w_h = self.W
            D = testmatrix.sum(axis=1)

            # compute hidden activations
            h = self.sigmoid(np.dot(testmatrix, w_vh) + np.outer(D, w_h))

            # compute visible activations
            v = np.dot(h, w_vh.T) + w_v
            pdf = self.softmax(v)

            # compute the per word perplexity
            z = np.nansum(testmatrix * np.log(pdf))
            s = np.sum(D)
            ppl = np.exp(-z / s)
            return ppl

        def approx_prob(self, dtm):
            w_vh, w_v, w_h = self.W
            D = dtm.sum(axis=1)
            # compute hidden activations
            h = self.sigmoid(np.dot(dtm, w_vh) + np.outer(D, w_h))

            # compute visible activations
            v = np.dot(h, w_vh.T) + w_v
            pdf = self.softmax(v)

            return pdf

        def ppl_exact_ais(
            self, testmatrix, S=10000, niter=100, MH_steps=0, D=[10, 20, 40, 60]
        ):
            """
            return the exact perplepxity
            given a document term matrix
            using Annealed Importance Sampling

            S: number of intermediate distributions
            niter: number of AIS runs
            MH_steps: number of MH steps per intermediate distribution (0 means just Gibbs sampling)
            D: list of document lengths to use for the AIS runs
            """
            log_Zb_list = []
            print("Estimating partition function using AIS...")
            for d in D:
                log_Zb, Za, log_avg_ratio, var_log_ratio = self.ais(
                    S=S, niter=niter, D=d, MH_steps=MH_steps
                )
                log_Zb_list.append(log_Zb)

            # estimate partition function for each document length
            slope, intercept = self.simple_linreg(
                X=np.array(D), Y=np.array(log_Zb_list)
            )

            N = testmatrix.shape[0]
            total_loglik = 0
            print("Computing exact perplexity...")
            for i in tqdm(range(N)):
                doc = testmatrix[i].reshape(1, -1)
                D = int(doc.sum())
                log_Zb = intercept + slope * D
                loglik = self.neg_free_energy(doc) - log_Zb
                total_loglik += loglik
            avg_loglik = total_loglik / N
            ppl = np.exp(-avg_loglik)
            return ppl

        def ais(self, S=1000, niter=100, D=20, MH_steps=0):
            """
            Annealed Importance Sampling to estimate the partition function of the RSM
            S: number of intermediate distributions
            niter: number of AIS runs
            D: document length for the AIS runs
            MH_steps: number of MH steps per intermediate distribution (0 means just Gibbs sampling)
            """

            T = self.hidden
            Za = 2**T
            K = self.visible  # voacb length
            # inverse temperature values
            beta = np.arange(start=0, stop=1 + 1 / S, step=1 / S)

            # intermediate pdf
            def temp_pdf(docvec, b):
                return np.exp(b * np.log(self.marginal_pdf(docvec)))

            def log_temp_pdf(docvec, b):
                return b * self.neg_free_energy_single_doc(docvec)
                # return b*np.log(self.marginal_pdf(docvec))

            log_w_ais_list = np.empty(niter)
            for it in tqdm(range(niter)):
                v_sampled = np.random.multinomial(D, np.ones(K) / K, size=1)[0]

                # loop
                log_w_ais = 0  # w_ais = 1
                for s in range(S - 1):
                    if MH_steps > 0:

                        def lpd(doc):
                            return log_temp_pdf(doc, beta[s])

                        for m in range(MH_steps):
                            v_sampled = self.MH_transition_vec(v_sampled, logpdf=lpd)
                    else:
                        v_sampled = self.gibbs_transition_vec(v_sampled)

                    logratio = log_temp_pdf(v_sampled, beta[s + 1]) - log_temp_pdf(
                        v_sampled, beta[s]
                    )
                    if not np.isnan(logratio):
                        log_w_ais = log_w_ais + logratio
                    # ratio = temp_pdf(v_sampled, beta[s+1])/temp_pdf(v_sampled, beta[s])
                    # w_ais = w_ais*ratio

                log_w_ais_list[it] = log_w_ais

            vec = log_w_ais_list - np.log(log_w_ais_list.shape[0])
            log_avg_ratio = np.max(vec) + np.log(np.sum(np.exp(vec - np.max(vec))))

            var_log_ratio = np.nanvar(log_w_ais_list)

            log_Zb = log_avg_ratio + np.log(Za)
            # Zb = np.exp(log_Zb)

            return log_Zb, Za, log_avg_ratio, var_log_ratio

        def simple_linreg(self, X, Y):
            """
            Simple linear regression to predict Y from X
            Returns coefficients and intercept
            """
            # Calculate means
            mean_x = np.mean(X)
            mean_y = np.mean(Y)

            # Calculate standard deviations
            sd_x = np.std(X, ddof=1)
            sd_y = np.std(Y, ddof=1)

            # Calculate correlation
            correlation = np.corrcoef(X, Y)[0, 1]

            # Calculate slope (b1) using the formula: b1 = (correlation * sd_y) / sd_X
            slope = (correlation * sd_y) / sd_x

            # Calculate intercept (b0) using the formula: b0 = mean_y - slope * mean_X
            intercept = mean_y - slope * mean_x

            return slope, intercept
