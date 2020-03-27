import json


# Retrieve the corpus and the labels
def retrieve(path_corpus,path_labels = ""):
    corpus = []
    labels = []
    with open(path_corpus) as file_input:
        for line in file_input:
            corpus.append(str(line))
    if len(path_labels)>1:
        with open(path_labels) as file_input:
            for label in file_input:
                labels.append(json.loads(label))
    result = {}
    result["corpus"] = corpus
    result["doc_labels"] = labels
    return result