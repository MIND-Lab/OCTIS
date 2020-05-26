from nltk.corpus import reuters


def retrieve():
    """
    Retrieve the corpus and the labels

    Returns
    -------
    result : dictionary with corpus, training and test partition
             and labels of the corpus
    """
    documents = reuters.fileids()
    corpus = []
    labels = []
    partition = []
    for document in documents:
        labels.append(reuters.categories(document))
        doc = reuters.words(document)
        partition.append(document.split("/")[0])
        doc_with_spaces = " ".join(doc)
        corpus.append(doc_with_spaces)
    result = {}
    result["partition"] = partition
    result["corpus"] = corpus
    result["doc_labels"] = labels
    return result
