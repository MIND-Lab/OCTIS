from pathlib import Path




'''
# MODELS PARAMETERS #
'''
model_descriptions = {'LDA': {'name': 'Latent Dirichlet Allocation (LDA)',
                              'citation': 'David M. Blei, Andrew Y. Ng, Michael I. Jordan: Latent Dirichlet Allocation. '
                                          'NIPS 2001: 601-608'},
                      'ETM': {'name': 'Embedded Topic Models (ETM)',
                              'citation': 'Adji Bousso Dieng, Francisco J. R. Ruiz, David M. Blei: Topic Modeling in '
                                          'Embedding Spaces. Trans. Assoc. Comput. Linguistics 8: 439-453 (2020)'},
                      'LSI': {'name': 'Latent Semantic Indexing',
                              'citation': 'Deerwester, S., et al, Improving Information Retrieval with Latent Semantic '
                                          'Indexing, Proceedings of the 51st Annual Meeting of the American Society for '
                                          'Information Science 25, 1988, pp. 36–40'},
                      'ProdLDA': {'name': 'Product-of-Experts LDA',
                                  'citation': 'Akash Srivastava, Charles Sutton: Autoencoding Variational Inference For '
                                              'Topic Models. ICLR (Poster) 2017'},
                      'NeuralLDA': {'name': 'Neural LDA',
                                    'citation': 'Akash Srivastava, Charles Sutton: Autoencoding Variational Inference For '
                                                'Topic Models. ICLR (Poster) 2017'},
                      'NMF': {'name': 'Non-negative Matrix Factorization',
                              'citation': 'Daniel D. Lee & H. Sebastian Seung (2001). Algorithms for Non-negative Matrix '
                                          'Factorization. Advances in Neural Information Processing Systems 13: '
                                          'Proceedings of the 2000 Conference. MIT Press. pp. 556–562.'},
                      'RSM' : {'name': 'Replicated Softmax Model', 
                               'citation':  'Ruslan R. Salakhutdinov, Geoffrey E. Hinton.' 
                                            'Replicated Softmax: An Undirected Topic Model.' 
                                            'Advances in neural information processing systems, 22 (2009).'},   

                      'oRSM' : {'name': 'Over Replicated Softmax Model',      
                                'citation': 'Srivastava, N., Salakhutdinov, R. R., & Hinton, G. E. (2013).' 
                                            'Modeling documents with deep boltzmann machines.' 
                                            'arXiv preprint arXiv:1309.6865.'   
                                          }
                        }

model_hyperparameters = {
    'LDA': {
        'alpha': {'type': 'Real', 'default_value': 0.1, 'min_value': 1e-4, 'max_value': 20,
                  'step': 1e-4, 'alternative_name': '&alpha;',
                  'description': 'symmetric Dirichlet prior controlling the document-topic distribution. '
                                 'Low values (<1) place more weight on having each document composed of '
                                 'only a few dominant topics (type: real)'},
        'eta': {'type': 'Real', 'default_value': 0.1, 'min_value': 1e-4, 'max_value': 20, 'step': 1e-4,
                'alternative_name': '&beta;', 'description': 'symmetric Dirichlet prior over the vocabulary. '
                                                             'Low values (<1) place more weight on having each topic'
                                                             ' composed of only a few dominant words (type: real)'},
        'num_topics': {'type': 'Integer', 'default_value': 10, 'min_value': 2,
                       'max_value': 200, 'step': 1, 'alternative_name': 'number of topics', 'description':
                           'number of topics to discover (integer)'},
        'passes': {'type': 'Integer', 'default_value': 1, 'min_value': 1, 'max_value': 10, 'step': 1,
                   'description': 'Number of passes through the corpus during training', 'alternative_name':
                       'Number of passes'},
        'iterations': {'type': 'Integer', 'default_value': 50, 'min_value': 5,
                       'max_value': 2000, 'step': 1,
                       'description': 'Number of iterations of the topic model algorithm (integer)'}},
    'ETM': {'num_topics': {'type': 'Integer', 'default_value': 10, 'min_value': 2,
                           'max_value': 200, 'step': 1, 'alternative_name': 'number of topics', 'description':
                               'number of topics to discover (integer)'},
            'num_epochs': {'type': 'Integer', 'default_value': 50, 'min_value': 5,
                           'max_value': 300, 'step': 1, 'alternative_name': 'number of training epochs', 'description':
                               'number of training epochs (type: integer)'},
            't_hidden_size':
                {'type': 'Categorical', 'default_value': 800,
                 'possible_values': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                 'alternative_name': 'hidden size',
                 'description': 'size of the deepest hidden layer (type: categorical)'},
            'rho_size':
                {'type': 'Categorical', 'default_value': 300, 'possible_values': [100, 200, 300, 400, 500],
                 'alternative_name': 'rho size',
                 'description': 'size of the word embeddings to learn (type: categorical)'},

            # 'embedding_size': {'type': 'Categorical', 'default_value': 300,
            #                  'alternative_name': 'pretrained embeddings size',
            #                  'description': 'size of the word embeddings that have been uploaded (type: categorical)'},

            'activation': {'type': 'Categorical', 'default_value': 'relu',
                           'possible_values': ['relu', 'tanh', 'sigmoid', 'softplus', 'rrelu', 'leakyrelu', 'elu',
                                               'selu', 'glu'],
                           'alternative_name': 'activation function',
                           'description': 'activation function applied to the hidden layer (type: categorical)'},
            'dropout': {'type': 'Real', 'default_value': 0.5, 'min_value': 0.0, 'max_value': 0.95, 'step': 0.05,
                        'description': 'Dropout applied to the hidden layer',
                        'alternative_name': 'dropout'},
            'lr': {'type': 'Real', 'default_value': 0.005, 'min_value': 1e-6, 'max_value': 0.1, 'step': 1e-6,
                   'description': 'Learning rate for the training of the network',
                   'alternative_name': 'learning rate'},
            'batch_size': {'type': 'Categorical', 'default_value': 200,
                           'possible_values': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                           'alternative_name': 'batch size',
                           'description': 'batch size (type: categorical)'},
            'bow_norm': {'type': 'Categorical', 'default_value': 1, 'possible_values': [0, 1],
                         'alternative_name': 'bow normalization', 'description':
                             'Whether to normalize (1) or not normalize (0) the BOW (bag-of-words) documents.'},
            'wdecay': {'type': 'Real', 'default_value': 1.2e-6, 'min_value': 0, 'max_value': 1, 'step': 1e-6,
                       'alternative_name': 'weight decay', 'description':
                           'Weight decay factor.'},

            'optimizer': {'type': 'Categorical', 'default_value': 'sgd',
                          'possible_values': ['adam', 'adagrad', 'adadelta', 'rmsprop', 'asgd', 'sgd'],
                          'alternative_name': 'optimizer',
                          'description': 'optimizer algorithm (type: categorical)'},

            },
    'LSI': {
        'decay': {'type': 'Real', 'default_value': 1.0, 'min_value': 0.0, 'max_value': 1.0, 'step': 0.1,
                  'description': 'Weight of existing observations relatively to new ones',
                  'alternative_name': 'decay'},
        'onepass': {'type': 'Categorical', 'default_value': True, 'possible_values': [True, False],
                    'description': 'Whether the one-pass algorithm should be used for training. '
                                   'Pass False to force a multi-pass stochastic algorithm.',
                    'alternative_name': 'one-pass algorithm'},
        'num_topics': {'type': 'Integer', 'default_value': 10, 'min_value': 2,
                       'max_value': 200, 'step': 1, 'alternative_name': 'number of topics', 'description':
                           'number of topics to discover'},
        'power_iters': {'type': 'Integer', 'default_value': 1, 'min_value': 1, 'max_value': 5, 'step': 1,
                        'description': 'Number of power iteration steps to be used. Increasing the number of power '
                                       'iterations improves accuracy, but lowers performance', 'alternative_name':
                        'Number of power iteration steps'},
        'extra_samples': {'type': 'Integer', 'default_value': 100, 'min_value': 0, 'max_value': 500, 'step': 1,
                          'description': 'Extra samples to be used besides the rank k. Can improve accuracy',
                          'alternative_name': 'Number of extra samples'}},
    'NMF': {
        'num_topics': {'type': 'Integer', 'default_value': 10, 'min_value': 2, 'max_value': 200, 'step': 1,
                       'alternative_name': 'number of topics', 'description': 'number of topics to discover'
                       },
        'passes': {'type': 'Integer', 'default_value': 1, 'min_value': 1, 'max_value': 10, 'step': 1,
                   'description': 'Number of passes through the corpus during training', 'alternative_name':
                       'Number of passes'},
        'normalize': {'type': 'Categorical', 'default_value': True, 'possible_values': [True, False],
                      'description': 'Whether to normalize the result'},
        'eval_every': {'type': 'Integer', 'default_value': 10, 'min_value': 1, 'max_value': 50, 'step': 1,
                       'description': 'Number of batches after which l2 norm of (v - Wh) is computed. '
                                      'Decreases performance if set too low.', 'alternative_name': 'evaluate every'},
        'kappa': {'type': 'Real', 'default_value': 1.0, 'min_value': 0.1, 'max_value': 5.0, 'step': 0.1,
                  'alternative_name': 'k',
                  'description': 'Gradient descent step size. Larger value makes the model train faster, but could lead to non-convergence if set too large.'},
        'w_max_iter': {'type': 'Integer', 'default_value': 200, 'min_value': 5, 'max_value': 1000, 'step': 1,
                       'alternative_name': 'Maximum iterations for W', 'description':
                           'Maximum number of iterations to train the matrix W per each batch (type: integer'},
        'h_max_iter': {'type': 'Integer', 'default_value': 50, 'min_value': 5, 'max_value': 1000, 'step': 1,
                       'alternative_name': 'Maximum iterations for H', 'description':
                           'Maximum number of iterations to train the matrix H per each batch (type: integer)'
                       },
        'w_stop_condition': {'type': 'Real', 'default_value': 0.0001, 'min_value': 1E-6, 'max_value': 0.1,
                             'step': 1E-6, 'alternative_name': 'stopping condition for W',
                             'description': 'If error difference gets less than that, training of W stops for '
                                            'the current batch. (type: real)'},
        'h_stop_condition': {'type': 'Real', 'default_value': 0.0001, 'min_value': 1E-6, 'max_value': 0.1,
                             'step': 1E-6, 'alternative_name': 'stopping condition for H',
                             'description': 'If error difference gets less than that, training of H stops for '
                                            'the current batch. (type: real)'}},
    'NMF_scikit': {
        'num_topics': {'type': 'Integer', 'default_value': 10, 'min_value': 2, 'max_value': 200, 'step': 1,
                       'alternative_name': 'number of topics', 'description': 'number of topics to discover'
                       },
        'init': {'type': 'Categorical', 'default_value': None,
                 'possible_values': [None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar']},
        'alpha': {'type': 'Real', 'default_value': 0.0, 'min_value': 0.0, 'max_value': 1.0, 'step': 0.1},
        'l1_ratio': {'type': 'Real', 'default_value': 0.0, 'min_value': 0, 'max_value': 1, 'step': 0.1}},
    'NeuralLDA': {
        'num_topics': {'type': 'Integer', 'default_value': 10, 'min_value': 2, 'max_value': 200, 'step': 1,
                       'alternative_name': 'number of topics', 'description': 'number of topics to discover'
                       },
        'activation': {'type': 'Categorical', 'default_value': 'softplus',
                       'possible_values': ['softplus', 'relu', 'sigmoid', 'swish', 'tanh', 'leakyrelu',
                                           'rrelu', 'elu', 'selu'],
                       'alternative_name': 'activation function',
                       'description': 'activation function applied to the hidden layer (type: categorical)'},
        'dropout': {'type': 'Real', 'default_value': 0.2, 'min_value': 0.0, 'max_value': 0.95, 'step': 0.05,
                    'description': 'Dropout applied to the hidden layer',
                    'alternative_name': 'dropout'},
        'batch_size': {'type': 'Categorical', 'default_value': 200,
                       'possible_values': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                       'alternative_name': 'batch size',
                       'description': 'batch size (type: categorical)'},
        'lr': {'type': 'Real', 'default_value': 2e-3, 'min_value': 1e-6, 'max_value': 0.1, 'step': 1e-6,
               'description': 'Learning rate for the training of the network',
               'alternative_name': 'learning rate'},
        'momentum': {'type': 'Real', 'default_value': 0.99, 'min_value': 0.01, 'max_value': 1, 'step': 0.01,
                     'description': 'Momentum',
                     'alternative_name': 'Momentum'},
        'solver': {'type': 'Categorical', 'default_value': 'sgd',
                   'possible_values': ['adagrad', 'adam', 'sgd', 'adadelta', 'rmsprop'],
                   'alternative_name': 'optimizer',
                   'description': 'optimizer algorithm (type: categorical)'},
        'num_epochs': {'type': 'Integer', 'default_value': 50, 'min_value': 5,
                       'max_value': 300, 'step': 1, 'alternative_name': 'number of training epochs', 'description':
                           'number of training epochs (type: integer)'},
        'num_neurons':
            {'type': 'Categorical', 'default_value': 800,
             'possible_values': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
             'alternative_name': 'Number of neurons',
             'description': 'Number of neurons for each layer of the network (type: integer)'},

        'num_layers':
            {'type': 'Categorical', 'default_value': 2,
             'possible_values': [1, 2, 3, 4, 5],
             'alternative_name': 'Number of layers',
             'description': 'Number of layers of the network (type: integer)'}},
    'ProdLDA': {
        'num_topics': {'type': 'Integer', 'default_value': 10, 'min_value': 2, 'max_value': 200, 'step': 1,
                       'alternative_name': 'number of topics', 'description': 'number of topics to discover'
                       },
        'activation': {'type': 'Categorical', 'default_value': 'softplus',
                       'possible_values': ['softplus', 'relu', 'sigmoid', 'swish', 'tanh', 'leakyrelu',
                                           'rrelu', 'elu', 'selu'],
                       'alternative_name': 'activation function',
                       'description': 'activation function applied to the hidden layer (type: categorical)'},
        'dropout': {'type': 'Real', 'default_value': 0.2, 'min_value': 0.0, 'max_value': 0.95, 'step': 0.05,
                    'description': 'Dropout applied to the hidden layer',
                    'alternative_name': 'dropout'},
        'batch_size': {'type': 'Categorical', 'default_value': 200,
                       'possible_values': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                       'alternative_name': 'batch size',
                       'description': 'batch size (type: categorical)'},
        'lr': {'type': 'Real', 'default_value': 2e-3, 'min_value': 1e-6, 'max_value': 0.1, 'step': 1e-6,
               'description': 'Learning rate for the training of the network',
               'alternative_name': 'learning rate'},
        'momentum': {'type': 'Real', 'default_value': 0.99, 'min_value': 0.01, 'max_value': 1, 'step': 0.01,
                     'description': 'Momentum',
                     'alternative_name': 'Momentum'},
        'solver': {'type': 'Categorical', 'default_value': 'sgd',
                   'possible_values': ['adagrad', 'adam', 'sgd', 'adadelta', 'rmsprop'],
                   'alternative_name': 'optimizer',
                   'description': 'optimizer algorithm (type: categorical)'},
        'num_epochs': {'type': 'Integer', 'default_value': 50, 'min_value': 5,
                       'max_value': 300, 'step': 1, 'alternative_name': 'number of training epochs', 'description':
                           'number of training epochs (type: integer)'},
        'num_neurons':
            {'type': 'Categorical', 'default_value': 800,
             'possible_values': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
             'alternative_name': 'Number of neurons',
             'description': 'Number of neurons for each layer of the network (type: integer)'},

        'num_layers':
            {'type': 'Categorical', 'default_value': 2,
             'possible_values': [1, 2, 3, 4, 5],
             'alternative_name': 'Number of layers',
             'description': 'Number of layers of the network (type: integer)'}
    }
}

'''
# METRIC PARAMETERS #
'''
metric_parameters = {
    # coherence
    "Coherence": {
        "name": "Coherence", "module": "coherence_metrics",
        "texts": {"type": "String", "default_value": "use dataset texts"},
        "topk": {"type": "Integer", "default_value": 10, "min_value": 5, "max_value": 30},
        "measure": {"type": "Categorical", "default_value": "c_npmi",
                    "possible_values": ['u_mass', 'c_v', 'c_uci', 'c_npmi']}},
    # diversity
    "TopicDiversity": {
        "name": "% Unique words",
        "module": "diversity_metrics",
        "topk": {"type": "Integer", "default_value": 10, "min_value": 5,
                 "max_value": 30}},

    "InvertedRBO": {"name": "IRBO", "module": "diversity_metrics",
                    "topk": {"type": "Integer", "default_value": 10, "min_value": 5,
                             "max_value": 30},
                    "weight": {"type": "Real", "default_value": 0.9, "min_value": 0.0,
                               "max_value": 1.0}},
    # divergences
    "KL_uniform": {"name": "KL-U", "module": "topic_significance_metrics",
                   "description": "KL-Uniform: It measures the distance (measured using the KL-divergence) "
                                  "of each topic-word distribution "
                                  "from the uniform distribution over the words. Significant topics are "
                                  "supposed to be skewed towards a few coherent and related words and "
                                  "distant from the uniform distribution."},
    "KL_background": {"name": "KL-B", "module": "topic_significance_metrics",
                      "description": "It measures the distance of a topic k to a “background” topic, which is a "
                                     "generic topic that is found equally probable in all the documents. Meaningful"
                                     " topics appear in a small subset of the data,thus higher values of KL–B are"
                                     " preferred"},
    "KL_vacuous": {"name": "KL-V", "module": "topic_significance_metrics",
                   "description": "It measures the distance between each topic-word distribution and the "
                                  "empirical word distribution of the whole dataset, also called “vacuous” "
                                  "distribution. The closer the word-topic distribution is to the empirical "
                                  "distribution of the sample, the less its significance is expected to be"},

    # classification
    "F1Score": {"name": "F1Score", "module": "classification_metrics",
                "dataset": {"type": "String", "default_value": "use selected dataset"},
                'average': {"type": "Categorical", "default_value": "micro",
                            "possible_values": ['binary', 'micro', None, 'macro', 'samples', 'weighted']
                            }}

}

'''
# OPTIMIZATION PARAMETERS #
'''

optimization_parameters = {
    "surrogate_models": [{"name": "Gaussian process", "id": "GP"},
                         {"name": "Random forest", "id": "RF"}],
    "acquisition_functions": [{"name": "Upper confidence bound", "id": "LCB"},
                              {"name": "Expected improvement", "id": "EI"}]
}

'''
# PARAMETERS INFO #
'''
HDP_hyperparameters_info = """
max_chunks (int, optional) – Upper bound on how many chunks to process. It wraps around corpus beginning in another corpus pass, if there are not enough chunks in the corpus. \n

max_time (int, optional) – Upper bound on time (in seconds) for which model will be trained. \n

chunksize (int, optional) – Number of documents in one chuck. \n

kappa (float,optional) – Learning parameter which acts as exponential decay factor to influence extent of learning from each batch. \n

tau (float, optional) – Learning parameter which down-weights early iterations of documents. \n

K (int, optional) – Second level truncation level \n

T (int, optional) – Top level truncation level \n

alpha (int, optional) – Second level concentration \n

gamma (int, optional) – First level concentration \n

eta (float, optional) – The topic Dirichlet \n

scale (float, optional) – Weights information from the mini-chunk of corpus to calculate rhot. \n

var_converge (float, optional) – Lower bound on the right side of convergence. Used when updating variational parameters for a single document.
"""

LDA_hyperparameters_info = """
num_topics (int, optional) – The number of requested latent topics to be extracted from the training corpus. \n

distributed (bool, optional) – Whether distributed computing should be used to accelerate training. \n

chunksize (int, optional) – Number of documents to be used in each training chunk. \n

passes (int, optional) – Number of passes through the corpus during training. \n

update_every (int, optional) – Number of documents to be iterated through for each update. Set to 0 for batch learning, > 1 for online iterative learning. \n

alpha ({numpy.ndarray, str}, optional) – Can be set to an 1D array of length equal to the number of expected topics that expresses our a-priori belief for the each topics’ probability. Alternatively default prior selecting strategies can be employed by supplying a string:

’asymmetric’: Uses a fixed normalized asymmetric prior of 1.0 / topicno.

’auto’: Learns an asymmetric prior from the corpus (not available if distributed==True). \n

eta ({float, np.array, str}, optional) – A-priori belief on word probability, this can be:

scalar for a symmetric prior over topic/word probability,

vector of length num_words to denote an asymmetric user defined probability for each word,

matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination,

the string ‘auto’ to learn the asymmetric prior from the data. \n

decay (float, optional) – A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten when each new document is examined. Corresponds to Kappa from Matthew D. Hoffman, David M. Blei, Francis Bach: “Online Learning for Latent Dirichlet Allocation NIPS’10”. \n

offset (float, optional) –

Hyper-parameter that controls how much we will slow down the first steps the first few iterations.
Corresponds to Tau_0 from Matthew D. Hoffman, David M. Blei, Francis Bach: “Online Learning for Latent Dirichlet Allocation NIPS’10”. \n

eval_every (int, optional) – Log perplexity is estimated every that many updates. Setting this to one slows
down training by ~2x. \n

iterations (int, optional) – Maximum number of iterations through the corpus when inferring the topic
distribution of a corpus. \n

gamma_threshold (float, optional) – Minimum change in the value of the gamma parameters to continue iterating. \n

minimum_probability (float, optional) – Topics with a probability lower than this threshold will be filtered out. \n

random_state ({np.random.RandomState, int}, optional) – Either a randomState object or a seed to generate one. Useful
for reproducibility. \n

minimum_phi_value (float, optional) – if per_word_topics is True, this represents a lower bound on the term
probabilities. \n

per_word_topics (bool) – If True, the model also computes a list of topics, sorted in descending order of most likely
topics for each word, along with their phi values multiplied by the feature length (i.e. word count). \n
"""

LSI_hyperparameters_info = """
num_topics (int, optional) – Number of requested factors (latent dimensions)\n

chunksize (int, optional) – Number of documents to be used in each training chunk. \n

decay (float, optional) – Weight of existing observations relatively to new ones. \n

distributed (bool, optional) – If True - distributed mode (parallel execution on several machines) will be used. \n

onepass (bool, optional) – Whether the one-pass algorithm should be used for training. Pass False to force a multi-pass
 stochastic algorithm. \n

power_iters (int, optional) – Number of power iteration steps to be used. Increasing the number of power iterations
improves accuracy, but lowers performance \n

extra_samples (int, optional) – Extra samples to be used besides the rank k. Can improve accuracy.
"""

NMF_gensim_hyperparameters_info = """
num_topics (int, optional) – Number of topics to extract. \n

chunksize (int, optional) – Number of documents to be used in each training chunk. \n

passes (int, optional) – Number of full passes over the training corpus. Leave at default passes=1 if your input is an
 iterator. \n

kappa (float, optional) – Gradient descent step size. Larger value makes the model train faster, but could lead to
non-convergence if set too large. \n

minimum_probability – If normalize is True, topics with smaller probabilities are filtered out. If normalize is False,
 topics with smaller factors are filtered out. If set to None, a value of 1e-8 is used to prevent 0s. \n

w_max_iter (int, optional) – Maximum number of iterations to train W per each batch. \n

w_stop_condition (float, optional) – If error difference gets less than that, training of W stops for the current
batch. \n

h_max_iter (int, optional) – Maximum number of iterations to train h per each batch. \n

h_stop_condition (float) – If error difference gets less than that, training of h stops for the current batch. \n

eval_every (int, optional) – Number of batches after which l2 norm of (v - Wh) is computed. Decreases performance if
set too low. \n

normalize (bool or None, optional) – Whether to normalize the result. \n

random_state ({np.random.RandomState, int}, optional) – Seed for random generator. Needed for reproducibility. \n
"""

NMF_scikit_hyperparameters_info = """
num_topics (int) – Number of topics to extract. \n

init (string, optional) – Method used to initialize the procedure. Default: None. Valid options:

    None: ‘nndsvd’ if n_components <= min(n_samples, n_features),
otherwise random.

    ‘random’: non-negative random matrices, scaled with:
sqrt(X.mean() / n_components)

    ‘nndsvd’: Nonnegative Double Singular Value Decomposition (NNDSVD)
initialization (better for sparseness)

    ‘nndsvda’: NNDSVD with zeros filled with the average of X
(better when sparsity is not desired)

    ‘nndsvdar’: NNDSVD with zeros filled with small random values
(generally faster, less accurate alternative to NNDSVDa for when sparsity is not desired) \n

alpha (double, optional) – Constant that multiplies the regularization terms. Set it to zero to have no regularization. \n

l1_ratio (double, optional) – The regularization mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an elementwise L2 penalty (aka Frobenius Norm). For l1_ratio = 1 it is an elementwise L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
"""



RSM_hyperparameters_info = """
num_topics (int, default=50) – Number of latent topics (hidden units) in the Replicated Softmax Model.

epochs (int, default=5) – Number of training epochs (full passes over the dataset).

btsz (int, default=100) – Mini-batch size used during training.

lr (float, default=0.01) – Learning rate for parameter updates.

momentum (float, default=0.1) – Momentum coefficient (used when train_optimizer='momentum').

K (int, default=1) – Number of Gibbs sampling steps for k-step Contrastive Divergence (K-CD).

softstart (float, default=0.001) – Scale for random initialization of weights (weights ~ N(0,1)*softstart).

decay (float, default=0) – Regularization coefficient. If >0, interaction penalty is applied (L1 or L2).

penalty_L1 (bool, default=False) – If True use L1 regularization; otherwise L2 is used.

penalty_local (bool, default=False) – If True apply penalty locally per-weight; otherwise apply a global penalty.

epochs_per_monitor (int, default=1) – Frequency (in epochs) to record monitoring metrics when monitor=True.

monitor (bool, default=False) – If True compute and store log-likelihood / perplexity during training.

persistent_cd (bool, default=False) – If True use persistent contrastive divergence (PCD) chains.

mean_field_cd (bool, default=True) – If True use mean-field contrastive divergence (mfcd) updates.

increase_cd (bool, default=False) – If True use gradual k-step CD (k increases across epochs).

increase_speed (float, default=0) – Controls speed of gradual increase of k when increase_cd is True.

cd_type (str, default='mfcd') – Type of contrastive-divergence algorithm. Common values: 'mfcd' (mean-field CD), 'kcd' (k-step CD), 'pcd' or 'persistent' (persistent CD), 'gradkcd' (gradual kcd).

train_optimizer (str, default='sgd') – Optimizer used for parameter updates. Options include: 'sgd', 'momentum', 'adagrad', 'rmsprop', 'adam', 'full' (full-batch), 'minibatch'.

logdtm (bool, default=False) – If True apply log(1+count) transform to the document-term matrix before training.

val_dtm (array or None, default=None) – Validation document-term matrix (used when training with partitions).

random_state (int or None, default=None) – Seed for numpy RNG for reproducible runs.

rms_decay (float, default=0.9) – RMSProp moving-average decay (used if train_optimizer='rmsprop').

adam_decay1 (float, default=0.9) – Adam first-moment decay (beta1).

adam_decay2 (float, default=0.999) – Adam second-moment decay (beta2).
"""



oRSM_hyperparameters_info = """
num_topics (int, default=50) – Number of latent topics (hidden units) in the Over Replicated Softmax Model.

epochs (int, default=5) – Number of training epochs (full passes over the dataset).

pretrain_epochs (int, default=1) – Number of initial epochs that run the pretraining (mean-field) phase.

btsz (int, default=100) – Mini-batch size used during training.

M (int, default=30) – Number of hidden multinomial units in the additional replicated softmax layer (over-replication factor).

lr (float, default=0.01) – Learning rate for parameter updates.

momentum (float, default=0.1) – Momentum coefficient (used when train_optimizer='momentum').

softstart (float, default=0.001) – Scale for random initialization of weights (weights ~ N(0,1)*softstart).

decay (float, default=0) – Regularization coefficient. If >0, interaction penalty is applied (L1 or L2).

penalty_L1 (bool, default=False) – If True use L1 regularization; otherwise L2 is used.

penalty_local (bool, default=False) – If True apply penalty locally per-weight; otherwise apply a global penalty.

cd_type (str, default='mfcd') – Type of contrastive-divergence algorithm (common values: 'mfcd' mean-field CD, 'kcd' k-step CD, 'pcd' persistent CD).

train_optimizer (str, default='sgd') – Optimizer used for parameter updates. Options include: 'sgd', 'momentum', 'adagrad', 'rmsprop', 'adam'.

rms_decay (float, default=0.9) – RMSProp moving-average decay (used if train_optimizer='rmsprop').

adam_decay1 (float, default=0.9) – Adam first-moment decay (beta1).

adam_decay2 (float, default=0.999) – Adam second-moment decay (beta2).

logdtm (bool, default=False) – If True apply log(1+count) transform to the document-term matrix before training.

val_dtm (array or None, default=None) – Validation document-term matrix (used when training with partitions).

epochs_per_monitor (int, default=1) – Frequency (in epochs) to record monitoring metrics when monitor=True.

monitor (bool, default=False) – If True compute and store monitoring metrics (e.g., perplexity) during training.

random_state (int or None, default=None) – Seed for numpy RNG for reproducible runs.

use_partitions (bool, default=True) – Whether the dataset partitions (train/test) are used (class attribute).

softstart (float, default=0.001) – Initial scale for weight initialization.

epsilon (float, default=0.01) – Convergence threshold used by mean-field updates (internal training parameter).

Notes:
- The model accepts a document-term matrix (dtm) as training input; many hyperparameters (e.g., M, btsz, lr, optimizer) influence training dynamics and convergence.
- Pretraining uses a simplified k-CD step (pretrain_epochs) before full training.
"""