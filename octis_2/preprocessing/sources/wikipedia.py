import json
from sklearn.model_selection import train_test_split


def retrieve_wikipedia(path):
    """
    Retrieve the corpus and the labels

    Parameters
    ----------
    path : path of the wikipedia dataset
           to retrieve

    Returns
    -------
    result : dictionary with corpus and 
             labels of the corpus
    """
    corpus = []
    labels = []
    with open(path) as file_input:
        for line in file_input:
            article = json.loads(line)
            corpus.append(article["text"])
            labels.append(article["labels"])

    train, test = train_test_split(range(len(corpus)),
                                   test_size=0.3,
                                   train_size=0.7,
                                   stratify=labels)

    partitioned_corpus = []
    partitioned_labels = []

    for doc in train:
        partitioned_corpus.append(corpus[doc])
        partitioned_labels.append(labels[doc])

    for doc in test:
        partitioned_corpus.append(corpus[doc])
        partitioned_labels.append(labels[doc])

    result = {}
    result["corpus"] = partitioned_corpus
    result["partition"] = len(train)
    result["doc_labels"] = partitioned_labels

    return result
