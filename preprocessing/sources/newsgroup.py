import gensim.downloader as gd


# Retrieve the corpus and the labels
def retrieve():
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
