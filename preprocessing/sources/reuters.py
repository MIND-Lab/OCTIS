from nltk.corpus import reuters


# Retrieve the dataset and the labels
def retrieve():
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

