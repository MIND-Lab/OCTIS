from urllib.request import urlopen
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
    result : dictionary with corpus and 
             labels of the corpus
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
    result = {}
    result["corpus"] = corpus
    result["doc_labels"] = labels
    return result
