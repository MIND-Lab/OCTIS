import json


def retrieve(corpus_path, labels_path=""):
    """
    Retrieve corpus and labels of a custom dataset
    given their path

    Parameters
    ----------
    corpus_path : path of the corpus
    labels_path : path of the labels document

    Returns
    -------
    result : dictionary with corpus and 
             optionally labels of the corpus
    """
    corpus = []
    labels = []
    with open(corpus_path) as file_input:
        for line in file_input:
            corpus.append(str(line))
    if len(labels_path) > 1:
        with open(labels_path) as file_input:
            for label in file_input:
                labels.append(json.loads(label))
    result = {}
    result["corpus"] = corpus
    result["doc_labels"] = labels
    return result
