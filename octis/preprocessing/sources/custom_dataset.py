import json


def retrieve(corpus_path, labels_path="", edges_path=""):
    """
    Retrieve corpus and labels of a custom dataset
    given their path

    Parameters
    ----------
    corpus_path : path of the corpus
    labels_path : path of the labels document

    Returns
    -------
    result : dictionary with corpus and 
             optionally labels of the corpus
    """
    result = {}
    corpus = []
    labels = []
    with open(corpus_path) as file_input:
        for line in file_input:
            corpus.append(str(line))
    result["corpus"] = corpus

    if len(labels_path) > 1:
        with open(labels_path) as file_input:
            for label in file_input:
                labels.append(json.loads(label))
        result["doc_labels"] = labels

    if len(edges_path) > 1:
        doc_ids = {}
        tmp_edges = []
        with open(edges_path) as file_input:
            for line in file_input:
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
        result["edges"] = edges_list

    return result
