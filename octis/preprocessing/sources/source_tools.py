from urllib.request import urlopen
from sklearn.model_selection import train_test_split
import re

def _retrieve(corpus_path, labels_path, edges_path):
    """
    Retrieve M10 or DBLP corpus and labels
    given their path

    Parameters
    ----------
    corpus_path : path of the corpus
    labels_path : path of the labels document
    edges_path : path of the adjacent neighbours document

    Returns
    -------
    result : dictionary with corpus, training and test partition,
             adjacent neighbours and labels of the corpus
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

    doc_ids = {}
    tmp_edges = []
    url = urlopen(edges_path)
    for line in url:
        neighbours = str(line)
        neighbours = neighbours[2:len(neighbours)-3]
        doc_ids[neighbours.split()[0]] = True
        tmp_edges.append(neighbours)

    edges_list = []

    for edges in tmp_edges:
        tmp_element = ""
        for edge in edges.split():
            if edge in doc_ids:
                tmp_element = tmp_element + edge + " "
        edges_list.append(tmp_element[0:len(tmp_element)-1])
    '''
    train, test = train_test_split(range(len(corpus)),
                                   test_size=0.3,
                                   train_size=0.7,
                                   stratify=labels)

    partitioned_corpus = []
    partitioned_labels = []
    partitioned_edges = []

    for doc in train:
        partitioned_corpus.append(corpus[doc])
        partitioned_labels.append(labels[doc])
        partitioned_edges.append(edges_list[doc])

    for doc in test:
        partitioned_corpus.append(corpus[doc])
        partitioned_labels.append(labels[doc])
        partitioned_edges.append(edges_list[doc])

    result = {}
    result["corpus"] = partitioned_corpus
    result["edges"] = partitioned_edges
    result["partition"] = len(train)
    result["doc_labels"] = partitioned_labels
    '''
    result = {}
    result["corpus"] = corpus
    result["edges"] = edges
    result["partition"] = len(corpus)
    result["doc_labels"] = labels
    return result
