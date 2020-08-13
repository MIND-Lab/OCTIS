from nltk.corpus import reuters


def retrieve_reuters():
    """
    Retrieve the corpus and the labels

    Returns
    -------
    result : dictionary with corpus, training and test partition
             and labels of the corpus
    """
    documents = reuters.fileids()
    train_corpus = []
    test_corpus = []
    train_labels = []
    test_labels = []
    for document in documents:
        doc_partition = document.split("/")[0]
        doc = reuters.words(document)
        doc_with_spaces = " ".join(doc)
        if doc_partition == "training":
            train_labels.append(reuters.categories(document))
            train_corpus.append(doc_with_spaces)
        else:
            test_labels.append(reuters.categories(document))
            test_corpus.append(doc_with_spaces)
    result = {}
    result["corpus"] = train_corpus + test_corpus
    result["partition"] = len(train_corpus)
    result["doc_labels"] = train_labels + test_labels
    result["info"] = {
        "name": "Reuters-21578, Distribution 1.0",
        "link": "https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz",
        "source": "https://www.nltk.org/",
        "info": "https://kdd.ics.uci.edu/databases/reuters21578/README.txt"
    }
    return result
