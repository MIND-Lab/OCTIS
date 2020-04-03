import json


def retrieve(path):
    """
    Retrieve the corpus and the labels

    Parameters
    ----------
    path : path of the wikipedia dataset
           to retrieve

    Returns
    -------
    result : dictionary with corpus and 
             labels of the corpus
    """
    corpus = []
    labels = []
    with open(path) as file_input:
        for line in file_input:
            article = json.loads(line)
            corpus.append(article["text"])
            labels.append(article["labels"])
    result = {}
    result["corpus"] = corpus
    result["doc_labels"] = labels
    return result
