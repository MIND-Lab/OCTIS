import gensim.downloader as gd


def retrieve():
    """
    Retrieve the corpus and the labels

    Returns
    -------
    result : dictionary with corpus, training and test partition
             and labels of the corpus
    """
    dataset = gd.load("20-newsgroups")
    corpus = []
    labels = []
    partition = []

    for data in dataset:
        corpus.append(data["data"])
        if data["set"] == "train":
            partition.append("training")
        else:
            partition.append(data["set"])
        labels.append([data["topic"]])
    result = {}
    result["partition"] = partition
    result["corpus"] = corpus
    result["doc_labels"] = labels
    return result
