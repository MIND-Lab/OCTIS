from sklearn.feature_extraction.text import CountVectorizer

from octis.models.model import AbstractModel
from octis.models.pytorchavitm import datasets
from octis.models.pytorchavitm.avitm import avitm_model


class AVITM(AbstractModel):

    def __init__(self, num_topics=10, model_type='prodLDA', activation='softplus',
                 dropout=0.2, learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99,
                 solver='adam', num_epochs=100, reduce_on_plateau=False, prior_mean=0.0,
                 prior_variance=None, num_layers=2, num_neurons=100, num_samples=10,
                 use_partitions=True):
        """
            :param num_topics : int, number of topic components, (default 10)
            :param model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
            :param num_layers : int, number of layers (default 2)
            :param num_neurons: int, number of neurons of each layer (default: 100)
            :param activation : string, 'softplus', 'relu', (default 'softplus')
            :param dropout : float, dropout to use (default 0.2)
            :param learn_priors : bool, make priors a learnable parameter (default True)
            :param batch_size : int, size of batch to use for training (default 64)
            :param lr : float, learning rate to use for training (default 2e-3)
            :param momentum : float, momentum to use for training (default 0.99)
            :param solver : string, optimizer 'adam' or 'sgd' (default 'adam')
            :param num_epochs : int, number of epochs to train for, (default 100)
            :param num_samples: int, number of times theta needs to be sampled (default: 10)
            :param reduce_on_plateau : bool, reduce learning rate by 10x on plateau of 10 epochs (default False)
        """
        super().__init__()
        self.hyperparameters['num_topics'] = num_topics
        self.hyperparameters['model_type'] = model_type
        self.hyperparameters['activation'] = activation
        self.hyperparameters['dropout'] = dropout
        self.hyperparameters['learn_priors'] = learn_priors
        self.hyperparameters['batch_size'] = batch_size
        self.hyperparameters['lr'] = lr
        self.hyperparameters['momentum'] = momentum
        self.hyperparameters['solver'] = solver
        self.hyperparameters['num_epochs'] = num_epochs
        self.hyperparameters['reduce_on_plateau'] = reduce_on_plateau
        self.hyperparameters["prior_mean"] = prior_mean
        self.hyperparameters["prior_variance"] = prior_variance
        self.hyperparameters["num_neurons"] = num_neurons
        self.hyperparameters["num_layers"] = num_layers
        self.hyperparameters["num_samples"] = num_samples

        hidden_sizes = tuple([num_neurons for _ in range(num_layers)])
        self.use_partitions = use_partitions
        self.hyperparameters['hidden_sizes'] = tuple(hidden_sizes)

    def train_model(self, dataset, hyperparameters=None, top_words=10):
        """
            :param dataset: list of sentences for training the model
            :param hyperparameters: dict, with the below information:

            num_topics : int, number of topic components, (default 10)
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
            num_samples: int, number of times theta needs to be sampled (default: 10)
            reduce_on_plateau : bool, reduce learning rate by 10x on plateau of 10 epochs (default False)
        """
        if hyperparameters is None:
            hyperparameters = {}
        self.set_params(hyperparameters)

        if self.use_partitions:
            train, validation, test = dataset.get_partitioned_corpus(use_validation=True)

            data_corpus_train = [' '.join(i) for i in train]
            data_corpus_test = [' '.join(i) for i in test]
            data_corpus_validation = [' '.join(i) for i in validation]

            self.vocab = dataset.get_vocabulary()
            x_train, x_test, x_valid, input_size = \
                self.preprocess(self.vocab, data_corpus_train, test=data_corpus_test,
                                validation=data_corpus_validation)
        else:
            self.vocab = dataset.get_vocabulary()
            data_corpus = [' '.join(i) for i in dataset.get_corpus()]
            x_train, input_size = self.preprocess(self.vocab, train=data_corpus)

        self.model = avitm_model.AVITM_model(
            input_size=input_size, num_topics=self.hyperparameters['num_topics'],
            model_type=self.hyperparameters['model_type'], hidden_sizes=self.hyperparameters['hidden_sizes'],
            activation=self.hyperparameters['activation'], dropout=self.hyperparameters['dropout'],
            learn_priors=self.hyperparameters['learn_priors'], batch_size=self.hyperparameters['batch_size'],
            lr=self.hyperparameters['lr'], momentum=self.hyperparameters['momentum'],
            solver=self.hyperparameters['solver'], num_epochs=self.hyperparameters['num_epochs'],
            reduce_on_plateau=self.hyperparameters['reduce_on_plateau'], num_samples=self.hyperparameters[
                'num_samples'], topic_prior_mean=self.hyperparameters["prior_mean"],
            topic_prior_variance=self.hyperparameters["prior_variance"]
        )

        if self.use_partitions:
            self.model.fit(x_train, x_valid)
            result = self.inference(x_test)
        else:
            self.model.fit(x_train, None)
            result = self.model.get_info()
        return result

    def set_params(self, hyperparameters):
        self.hyperparameters['num_topics'] = \
            int(hyperparameters.get('num_topics', self.hyperparameters['num_topics']))
        self.hyperparameters['model_type'] = \
            hyperparameters.get('model_type', self.hyperparameters['model_type'])
        self.hyperparameters['activation'] = \
            hyperparameters.get('activation', self.hyperparameters['activation'])
        self.hyperparameters['dropout'] = float(hyperparameters.get('dropout', self.hyperparameters['dropout']))
        self.hyperparameters['learn_priors'] = \
            hyperparameters.get('learn_priors', self.hyperparameters['learn_priors'])
        self.hyperparameters['batch_size'] = \
            int(hyperparameters.get('batch_size', self.hyperparameters['batch_size']))
        self.hyperparameters['lr'] = float(hyperparameters.get('lr', self.hyperparameters['lr']))
        self.hyperparameters['momentum'] = \
            float(hyperparameters.get('momentum', self.hyperparameters['momentum']))
        self.hyperparameters['solver'] = hyperparameters.get('solver', self.hyperparameters['solver'])
        self.hyperparameters['num_epochs'] = \
            int(hyperparameters.get('num_epochs', self.hyperparameters['num_epochs']))
        self.hyperparameters['reduce_on_plateau'] = \
            hyperparameters.get('reduce_on_plateau', self.hyperparameters['reduce_on_plateau'])
        self.hyperparameters["prior_mean"] = \
            hyperparameters.get('prior_mean', self.hyperparameters['prior_mean'])
        self.hyperparameters["prior_variance"] = \
            hyperparameters.get('prior_variance', self.hyperparameters['prior_variance'])

        self.hyperparameters["num_layers"] = \
            int(hyperparameters.get('num_layers', self.hyperparameters['num_layers']))
        self.hyperparameters["num_neurons"] = \
            int(hyperparameters.get('num_neurons', self.hyperparameters['num_neurons']))

        self.hyperparameters['hidden_sizes'] = tuple(
            [self.hyperparameters["num_neurons"] for _ in range(self.hyperparameters["num_layers"])])

    def inference(self, x_test):
        assert isinstance(self.use_partitions, bool) and self.use_partitions
        results = self.model.predict(x_test)
        return results

    def partitioning(self, use_partitions=False):
        self.use_partitions = use_partitions

    @staticmethod
    def preprocess(vocab, train, test=None, validation=None):
        vocab2id = {w: i for i, w in enumerate(vocab)}
        vec = CountVectorizer(vocabulary=vocab2id, token_pattern=r'(?u)\b\w+\b')
        entire_dataset = train.copy()
        if test is not None:
            entire_dataset.extend(test)
        if validation is not None:
            entire_dataset.extend(validation)

        vec.fit(entire_dataset)
        idx2token = {v: k for (k, v) in vec.vocabulary_.items()}
        X_train = vec.transform(train)
        train_data = datasets.BOWDataset(X_train.toarray(), idx2token)
        input_size = len(idx2token.keys())

        if test is not None and validation is not None:
            x_test = vec.transform(test)
            test_data = datasets.BOWDataset(x_test.toarray(), idx2token)
            x_valid = vec.transform(validation)
            valid_data = datasets.BOWDataset(x_valid.toarray(), idx2token)
            return train_data, test_data, valid_data, input_size
        if test is None and validation is not None:
            x_valid = vec.transform(validation)
            valid_data = datasets.BOWDataset(x_valid.toarray(), idx2token)
            return train_data, valid_data, input_size
        if test is not None and validation is None:
            x_test = vec.transform(test)
            test_data = datasets.BOWDataset(x_test.toarray(), idx2token)
            return train_data, test_data, input_size
        if test is None and validation is None:
            return train_data, input_size
