from pathlib import Path


def _load_default_texts():
    """
    Loads default wikipedia texts

    Returns
    -------
    result : default wikipedia texts
    """
    file_name = "preprocessed_datasets/wikipedia/wikipedia_not_lemmatized_5/corpus.txt"
    result = []
    file = Path(file_name)
    if file.is_file():
        with open(file_name, 'r') as corpus_file:
            for line in corpus_file:
                result.append(line.split())
        return result
    return False


em_coherence = {
    'texts': _load_default_texts(),
    'topk': 10,
    'measure': 'c_npmi'
}

em_coherence_we = {
    'topk': 10,
    'word2vec_path': None,
    'binary': False
}

em_coherence_we_pc = {
    'topk': 10,
    'w2v_model': None
}

em_topic_diversity = {'topk': 10}

models_HDP_hyperparameters = {
    'corpus': None,
    'id2word': None,
    'max_chunks': None,
    'max_time': None,
    'chunksize': 256,
    'kappa': 1.0,
    'tau': 64.0,
    'K': 15,
    'T': 150,
    'alpha': 1,
    'gamma': 1,
    'eta': 0.01,
    'scale': 1.0,
    'var_converge': 0.0001,
    'outputdir': None,
    'random_state': None}

models_LDA_hyperparameters = {
    'corpus': None,
    'num_topics': 100,
    'id2word': None,
    'distributed': False,
    'chunksize': 2000,
    'passes': 1,
    'update_every': 1,
    'alpha': 'symmetric',
    'eta': None,
    'decay': 0.5,
    'offset': 1.0,
    'eval_every': 10,
    'iterations': 50,
    'gamma_threshold': 0.001,
    'minimum_probability': 0.01,
    'random_state': None,
    'ns_conf': None,
    'minimum_phi_value': 0.01,
    'per_word_topics': False,
    'callbacks': None}

models_LSI_hyperparameters = {
    'corpus': None,
    'num_topics': 100,
    'id2word': None,
    'distributed': False,
    'chunksize': 20000,
    'decay': 1.0,
    'onepass': True,
    'power_iters': 2,
    'extra_samples': 100}

models_NMF_hyperparaeters = {
    'corpus': None,
    'num_topics': 100,
    'id2word': None,
    'chunksize': 2000,
    'passes': 1,
    'kappa': 1.0,
    'minimum_probability': 0.01,
    'w_max_iter': 200,
    'w_stop_condition': 0.0001,
    'h_max_iter': 50,
    'h_stop_condition': 0.001,
    'eval_every': 10,
    'normalize': True,
    'random_state': None}
