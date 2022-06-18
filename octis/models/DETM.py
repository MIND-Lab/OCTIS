from octis.models.DETM_model import detm
from octis.models.base_etm import BaseETM
from octis.models.DETM_model import data
import torch
import warnings

class DETM(BaseETM):
    def __init__(self, num_topics=50, rho_size=300, embedding_size=300, t_hidden_size=800,
                 activation='relu', train_embeddings=1, eta_nlayers=3, eta_hidden_size=200,
                 delta=0.005, device='cpu', lr_factor=4.0, lr=0.005, anneal_lr=1, batch_size=100,
                 num_epochs=100, seed=2019, dropout=0.0, eta_dropout=0.0, clip=0.0,
                 nonmono=10, optimizer='adam', wdecay=1.2e-6, embeddings_path="", use_partitions=True):

        warnings.simplefilter('always', Warning)
        warnings.warn("Don't use this because it doesn't work :)",
                      Warning)

        super(DETM, self).__init__()
        self.hyperparameters = dict()
        self.hyperparameters['num_topics'] = int(num_topics)
        self.hyperparameters['num_epochs'] = int(num_epochs)
        self.hyperparameters['t_hidden_size'] = int(t_hidden_size)
        self.hyperparameters['rho_size'] = int(rho_size)
        self.hyperparameters['embedding_size'] = int(embedding_size)
        self.hyperparameters['activation'] = activation
        self.hyperparameters['eta_nlayers'] = eta_nlayers
        self.hyperparameters['eta_hidden_size'] = eta_hidden_size
        self.hyperparameters['delta'] = delta
        self.hyperparameters['dropout'] = float(dropout)
        self.hyperparameters['lr'] = float(lr)
        self.hyperparameters['lr_factor'] = float(lr)
        self.hyperparameters['anneal_lr'] = float(anneal_lr)
        self.hyperparameters['optimizer'] = optimizer
        self.hyperparameters['batch_size'] = int(batch_size)
        self.hyperparameters['clip'] = float(clip)
        self.hyperparameters['wdecay'] = float(wdecay)
        self.hyperparameters['eta_dropout'] = float(eta_dropout)
        self.hyperparameters['seed'] = int(seed)
        self.hyperparameters['clip'] = int(clip)
        self.hyperparameters['nonmono'] = int(nonmono)
        self.hyperparameters['train_embeddings'] = bool(train_embeddings)
        self.hyperparameters['embeddings_path'] = embeddings_path
        self.device = device
        self.early_stopping = None
        # TODO:  this we need to agree on this
        self.test_tokens, self.test_counts = None, None
        self.valid_tokens, self.valid_counts = None, None
        self.train_tokens, self.train_counts, self.vocab = None, None, None
        self.use_partitions = use_partitions
        self.model = None
        self.optimizer = None
        self.embeddings = None

        # are they enough or we need more

    def set_model(self, dataset, hyperparameters):
        if self.use_partitions:
            train_data, validation_data, testing_data = dataset.get_partitioned_corpus(use_validation=True)

            data_corpus_train = [' '.join(i) for i in train_data]
            data_corpus_test = [' '.join(i) for i in testing_data]
            data_corpus_val = [' '.join(i) for i in validation_data]

            vocab = dataset.get_vocabulary()
            self.vocab = {i: w for i, w in enumerate(vocab)}
            vocab2id = {w: i for i, w in enumerate(vocab)}

            self.train_tokens, self.train_counts, self.test_tokens, self.test_counts, self.valid_tokens, \
            self.valid_counts = self.preprocess(vocab2id, data_corpus_train, data_corpus_test, data_corpus_val)

        else:
            data_corpus = [' '.join(i) for i in dataset.get_corpus()]
            vocab = dataset.get_vocabulary()
            self.vocab = {i: w for i, w in enumerate(vocab)}
            vocab2id = {w: i for i, w in enumerate(vocab)}

            self.train_tokens, self.train_counts = self.preprocess(vocab2id, data_corpus, None)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.set_default_hyperparameters(hyperparameters)
        self.load_embeddings()
        ## define model and optimizer
        self.model = detm.DETM(num_topics=self.hyperparameters['num_topics'],
                               num_times=self.hyperparameters['num_times'],
                               vocab_size=len(self.vocab.keys()),
                               t_hidden_size=int(self.hyperparameters['t_hidden_size']),
                               eta_hidden_size=int(self.hyperparameters['eta_hidden_size']),
                               rho_size=int(self.hyperparameters['rho_size']),
                               emb_size=int(self.hyperparameters['embedding_size']),
                               theta_act=self.hyperparameters['activation'],
                               eta_nlayers=self.hyperparameters['eta_nlayers'],
                               delta=self.hyperparameters['eta_nlayers'],
                               embeddings=self.embeddings,
                               train_embeddings=self.hyperparameters['train_embeddings'],
                               enc_drop=self.hyperparameters['dropout']).to(self.device)
        print('model: {}'.format(self.model))

        self.optimizer = self.set_optimizer()

    def _train_epoch(self, epoch):
        """
        Train the model for the given epoch
        """
        # change to the way we are loading data the correct form .. @ask sylvia
        train_data_with_time = None
        train_data, train_times = data.get_time_columns(train_data_with_time)
        self.train_rnn_inp = data.get_rnn_input(
            self.train_tokens, self.train_counts, train_times, self.hyperparameters['num_times'], len(self.vocab),
            len(self.train_tokens))

        self.model.train()
        acc_loss = 0
        acc_nll = 0
        acc_kl_theta_loss = 0
        acc_kl_eta_loss = 0
        acc_kl_alpha_loss = 0
        cnt = 0
        indices = torch.randperm(train_data.shape[0])
        indices = torch.split(indices, self.hyperparameters['batch_size'])
        optimizer = self.set_optimizer()
        for idx, ind in enumerate(indices):
            optimizer.zero_grad()
            self.zero_grad()
            data_batch, times_batch = data.get_batch(train_data, ind, self.device,
                                                     train_times)  # we can use pytorch data loader here
            ### I comment the following row just because I need to make the code compile :/
            # times_batch = get_indices(train_times, times_batch)
            sums = data_batch.sum(1).unsqueeze(1)
            times_batch = torch.from_numpy(times_batch)
            if self.hyperparameters['bow_norm']:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            loss, nll, kl_alpha, kl_eta, kl_theta = self.model.forward(
                data_batch, normalized_data_batch, times_batch, self.train_rnn_inp, train_data.shape[0])
            loss.backward()
            if self.hyperparameters['clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.hyperparameters['clip'])
            optimizer.step()
            acc_loss += torch.sum(loss).item()
            acc_nll += torch.sum(nll).item()
            acc_kl_theta_loss += torch.sum(kl_theta).item()
            acc_kl_eta_loss += torch.sum(kl_eta).item()
            acc_kl_alpha_loss += torch.sum(kl_alpha).item()
            cnt += 1
            if idx % self.hyperparameters['log_interval'] == 0 and idx > 0:
                cur_loss = round(acc_loss / cnt, 2)
                cur_nll = round(acc_nll / cnt, 2)
                cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
                cur_kl_eta = round(acc_kl_eta_loss / cnt, 2)
                cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2)
                lr = optimizer.param_groups[0]['lr']
                print(
                    'Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. NELBO: {}'.format(
                        epoch, idx, len(indices), lr, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_loss))
        cur_loss = round(acc_loss / cnt, 2)
        cur_nll = round(acc_nll / cnt, 2)
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
        cur_kl_eta = round(acc_kl_eta_loss / cnt, 2)
        cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2)
        lr = optimizer.param_groups[0]['lr']
        print('*' * 100)
        print(
            'Epoch----->{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, lr, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_loss))
        print('*' * 100)
