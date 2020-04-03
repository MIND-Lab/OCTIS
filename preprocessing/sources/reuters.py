from nltk.corpus import reuters


def retrieve():
    """
    Retrieve the corpus and the labels

    Returns
    -------
    result : dictionary with corpus and 
             labels of the corpus
    """
    documents = reuters.fileids()
    corpus = []
    labels = []
    for document in documents:
        labels.append(reuters.categories(document))
        doc = reuters.words(document)
        doc_with_spaces = " ".join(doc)
        corpus.append(doc_with_spaces)
    result = {}
    result["corpus"] = corpus
    result["doc_labels"] = labels
    return result
