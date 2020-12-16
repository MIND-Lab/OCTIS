import os
import json
from pathlib import Path

# get the path to the framework folder
path = Path(os.path.dirname(os.path.realpath(__file__)))
path = str(path.parent)


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
    f = open(str(os.path.join(
        path, "preprocessed_datasets", datasetName, "metadata.json")),)
    data = json.load(f)
    return data


def getDocPreview(datasetName, documentNumber):
    datasetPath = str(os.path.join(
        path, "preprocessed_datasets", datasetName, "corpus.txt"))
    corpus = []
    file = Path(datasetPath)
    if file.is_file():
        with open(datasetPath, 'r') as corpus_file:
            for line in corpus_file:
                corpus.append(line)
    splitted = corpus[documentNumber].split()
    if len(splitted) > 40:
        return " ".join(splitted[0:40])
    return corpus[documentNumber]


def getVocabulary(path):
    vocabulary_file = open(path,)
    return json.load(vocabulary_file)
