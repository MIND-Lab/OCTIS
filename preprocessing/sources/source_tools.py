from urllib.request import urlopen
from sklearn.model_selection import train_test_split

import re


def _retrieve(corpus_path, labels_path):
    """
    Retrieve M10 or dblp corpus and labels
    given their path

    Parameters
    ----------
    corpus_path : path of the corpus
    labels_path : path of the labels document

    Returns
    -------
    result : dictionary with corpus, training and test partition
             and labels of the corpus
    """
    corpus = []
    url = urlopen(corpus_path)
    for line in url:
        corpus.append(str(line))
    labels = []
    url = urlopen(labels_path)
    for line in url:
        line = re.sub("[^.0-9\\s]", '', str(line))
        label_list = line.split()
        labels.append(label_list[1:len(label_list)])

    train, test = train_test_split(range(len(corpus)),
                                   test_size=0.3,
                                   train_size=0.7,
                                   stratify=labels)

    partition = ["train"] * len(corpus)
    for doc in train:
        partition[doc] = "training"

    for doc in test:
        partition[doc] = "test"

    result = {}
    result["corpus"] = corpus
    result["partition"] = partition
    result["doc_labels"] = labels

    return result
