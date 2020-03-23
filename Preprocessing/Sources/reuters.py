from nltk.corpus import reuters


def retrieve():
    documents = reuters.fileids()
    corpus = []
    labels = []
    for document in documents:
        labels.append(reuters.categories(document))
        doc = reuters.words(document)
        docWithSpaces = " ".join(doc)
        corpus.append(docWithSpaces)
    result = {}
    result["corpus"] = corpus
    result["docLabels"] = labels
    return result



