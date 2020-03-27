import json

# Retrieve the dataset and the labels
def retrieve(path):
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
