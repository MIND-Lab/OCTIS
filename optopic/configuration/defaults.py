from pathlib import Path


def _load_default_texts():
    """
    Loads default general texts

    Returns
    -------
    result : default wikipedia texts
    """
    file_name = "preprocessed_datasets/20newsgroup_validation/corpus.txt"
    result = []
    file = Path(file_name)
    if file.is_file():
        with open(file_name, 'r') as corpus_file:
            for line in corpus_file:
                result.append(line.split())
        return result
    return False


'''
# METRICS PARAMETERS #
'''
em_coherence = {'texts': _load_default_texts(), 'topk': 10,
                'measure': 'c_npmi'}
em_coherence_we = {'topk': 10, 'word2vec_path': None, 'binary': False}
em_coherence_we_pc = {'topk': 10, 'w2v_model': None}
em_topic_diversity = {'topk': 10}
em_invertedRBO = {'topk': 10, 'weight': 0.9}
em_word_embeddings_invertedRBO = {'topk': 10, 'weight': 0.9}
em_f1_score = {'average': 'micro'}

'''
# MODELS PARAMETERS #
'''

model_hyperparameters = {
    'LDA': {
        'alpha': {'type': 'Real', 'default_value': 0.1, 'min_value': 1e-4, 'max_value': 20},
        'eta': {'type': 'Real', 'default_value': 0.1, 'min_value': 1e-4, 'max_value': 20},
        'num_topics':  {'type': 'Integer', 'default_value': 10, 'min_value': 2,
                        'max_value': 200},
        'passes': {'type': 'Integer', 'default_value': 1, 'min_value': 1, 'max_value': 10},
        'iterations': {'type': 'Integer', 'default_value': 50, 'min_value': 5,
                       'max_value': 2000}},
    'ETM': {},
    'LSI': {
        'decay': {'type': 'Real', 'default_value': 1.0, 'min_value': 0.0, 'max_value': 1.0},
        'onepass': {'type': 'Categorical', 'default_value': True, 'possible_values': [True, False]},
        'num_topics':  {'type': 'Integer', 'default_value': 10, 'min_value': 2,
                        'max_value': 200},
        'power_iters': {'type': 'Integer', 'default_value': 1, 'min_value': 1, 'max_value': 5},
        'extra_samples': {'type': 'Integer', 'default_value': 100, 'min_value': 0, 'max_value': 500}},
    'NMF': {
        'num_topics':  {'type': 'Integer', 'default_value': 10, 'min_value': 2, 'max_value': 200},
        'passes': {'type': 'Integer', 'default_value': 1, 'min_value': 1, 'max_value': 10},
        'normalize': {'type': 'Categorical', 'default_value': True, 'possible_values': [True, False]},
        'eval_every': {'type': 'Integer', 'default_value': 10, 'min_value': 1, 'max_value': 50},
        'kappa': {'type': 'Real', 'default_value': 1.0, 'min_value': 0.1, 'max_value': 5.0},
        'w_max_iter': {'type': 'Integer', 'default_value': 200, 'min_value': 10, 'max_value': 1000},
        'h_max_iter': {'type': 'Integer', 'default_value': 50, 'min_value': 10, 'max_value': 1000},
        'w_stop_condition': {'type': 'Real', 'default_value': 0.0001, 'min_value': 1E-6, 'max_value': 0.1},
        'h_stop_condition': {'type': 'Real', 'default_value': 0.0001, 'min_value': 1E-6, 'max_value': 0.1}},
    '...': {}}



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
    "Topic_diversity": {
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
    "KL_uniform": {"name": "KL-U", "module": "topic_significance_metrics"},
    "KL_background": {"name": "KL-B", "module": "topic_significance_metrics"},
    "KL_vacuous": {"name": "KL-V", "module": "topic_significance_metrics"}
}


'''
# OPTIMIZATION PARAMETERS #
'''

optimization_parameters = {
    "surrogate_models": [{"name": "Gaussian proccess", "id": "GP"},
                         {"name": "Random forest", "id": "RF"},
                         {"name": "Random search", "id": "RS"}],
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

Hyper-parameter that controls how much we will slow down the first steps the first few iterations. Corresponds to Tau_0 from Matthew D. Hoffman, David M. Blei, Francis Bach: “Online Learning for Latent Dirichlet Allocation NIPS’10”. \n

eval_every (int, optional) – Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x. \n

iterations (int, optional) – Maximum number of iterations through the corpus when inferring the topic distribution of a corpus. \n

gamma_threshold (float, optional) – Minimum change in the value of the gamma parameters to continue iterating. \n

minimum_probability (float, optional) – Topics with a probability lower than this threshold will be filtered out. \n

random_state ({np.random.RandomState, int}, optional) – Either a randomState object or a seed to generate one. Useful for reproducibility. \n

minimum_phi_value (float, optional) – if per_word_topics is True, this represents a lower bound on the term probabilities. \n

per_word_topics (bool) – If True, the model also computes a list of topics, sorted in descending order of most likely topics for each word, along with their phi values multiplied by the feature length (i.e. word count). \n
"""

LSI_hyperparameters_info = """
num_topics (int, optional) – Number of requested factors (latent dimensions)\n

chunksize (int, optional) – Number of documents to be used in each training chunk. \n

decay (float, optional) – Weight of existing observations relatively to new ones. \n

distributed (bool, optional) – If True - distributed mode (parallel execution on several machines) will be used. \n

onepass (bool, optional) – Whether the one-pass algorithm should be used for training. Pass False to force a multi-pass stochastic algorithm. \n

power_iters (int, optional) – Number of power iteration steps to be used. Increasing the number of power iterations improves accuracy, but lowers performance \n

extra_samples (int, optional) – Extra samples to be used besides the rank k. Can improve accuracy.
"""

NMF_gensim_hyperparameters_info = """
num_topics (int, optional) – Number of topics to extract. \n

chunksize (int, optional) – Number of documents to be used in each training chunk. \n

passes (int, optional) – Number of full passes over the training corpus. Leave at default passes=1 if your input is an iterator. \n

kappa (float, optional) – Gradient descent step size. Larger value makes the model train faster, but could lead to non-convergence if set too large. \n

minimum_probability – If normalize is True, topics with smaller probabilities are filtered out. If normalize is False, topics with smaller factors are filtered out. If set to None, a value of 1e-8 is used to prevent 0s. \n

w_max_iter (int, optional) – Maximum number of iterations to train W per each batch. \n

w_stop_condition (float, optional) – If error difference gets less than that, training of W stops for the current batch. \n

h_max_iter (int, optional) – Maximum number of iterations to train h per each batch. \n

h_stop_condition (float) – If error difference gets less than that, training of h stops for the current batch. \n

eval_every (int, optional) – Number of batches after which l2 norm of (v - Wh) is computed. Decreases performance if set too low. \n

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
