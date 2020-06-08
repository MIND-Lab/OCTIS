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
