import os
import json
from pathlib import Path

# get the path to the framework folder
path = Path(os.path.dirname(os.path.realpath(__file__)))
path = str(path.parent.parent)


def scanDatasets():
    """
    Retrieves the name of each dataset present in the framework

    Returns
    -------
    res : list with name of each dataset as element
    """

    datasets = os.listdir(str(os.path.join(path, "preprocessed_datasets")))
    datasets.remove("README.txt")
    return datasets


def getDatasetMetadata(datasetName):
    """
    Retrieves the dataset metadata

    Returns
    -------
    data : dict with metadata if dataset is found, False otherwise
    """
    file = str(os.path.join(
        path, "preprocessed_datasets", datasetName, "corpus.tsv"))
    if os.path.isfile(file):
        f = open(file,)
        return {"total_documents":sum(1 for line in f)}
    return False


def getDocPreview(datasetName, documentNumber):
    """
    Retrieve the first 40 words of the selected document

    Parameters
    ----------
    datasetName: name of the dataset in which the document is located 
                 (the dataset must be in the preprocessed_datasets folder)
    documentNumber: number of the document to retrieve

    Returns
    out: First 40 words in the document
    """
    datasetPath = str(os.path.join(
        path, "preprocessed_datasets", datasetName, "corpus.tsv"))
    corpus = []
    if os.path.isfile(datasetPath):
        with open(datasetPath, 'r') as corpus_file:
            for line in corpus_file:
                corpus.append(line.split("\t")[0])
        splitted = corpus[documentNumber].split()
        if len(splitted) > 40:
            return " ".join(splitted[0:40])
        return corpus[documentNumber]
    return False




def getVocabulary(path):
    """
    Retrieves the vocabulary from the vocabulary file of an ezxperiment

    Returns
    -------
    vocabulary : a dictionary with id as a key and word as value,
                 returns False if the vocabulary is not found
    """
    if Path(path).is_file():
        vocabulary_file = open(path,)
        return json.load(vocabulary_file)
    return False


#def getVocabulary(path):
#    """
#    Retrieves the vocabulary from the vocabulary file of an ezxperiment
#
#    Returns
#    -------
#    vocabulary : a dictionary with id as a key and word as value,
#                 returns False if the vocabulary is not found
#    """
#    count = 0
#    vocabulary = {}
#    if Path(path).is_file():
#        with open(path, 'r') as f:
#            for line in f:
#                vocabulary[str(count)] = line.strip()
#                count+= 1  
#        return vocabulary      
#    return False
