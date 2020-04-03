import gensim.downloader as gd


def retrieve():
    """
    Retrieve the corpus and the labels

    Returns
    -------
    result : dictionary with corpus and 
             labels of the corpus
    """
    dataset = gd.load("20-newsgroups")
    corpus = []
    labels = []
    for data in dataset:
        corpus.append(data["data"])
        labels.append([data["topic"]])
    result = {}
    result["corpus"] = corpus
    result["doc_labels"] = labels
    return result
