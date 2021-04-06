import os
import json
from pathlib import Path

# get the path to the framework folder
path = Path(os.path.dirname(os.path.realpath(__file__)))
path = str(path.parent.parent)


def scanDatasets():
    """
    Retrieves the name of each dataset present in the framework

    :return: list with name of each dataset as element
    :rtype: List
    """

    datasets = os.listdir(str(os.path.join(path, "preprocessed_datasets")))
    datasets.remove("README.rst")
    return datasets


def getDatasetMetadata(datasetName):
    """
    Retrieves the dataset metadata

    :param datasetName: name of the dataset
    :type datasetName: String

    :return: dict with metadata if dataset is found, False otherwise
    :rtype: Dict
    """
    file = str(os.path.join(
        path, "preprocessed_datasets", datasetName, "corpus.tsv"))
    if os.path.isfile(file):
        f = open(file,)
        return {"total_documents": sum(1 for line in f)}
    return False


def getDocPreview(datasetName, documentNumber):
    """
    Retrieve the first 40 words of the selected document

    :param datasetName: name of the dataset in which the document is located 
                 (the dataset must be in the preprocessed_datasets folder)
    :type datasetName: String
    :param documentNumber: number of the document to retrieve
    :type documentNumber: Int

    :return: First 40 words in the document
    :rtype: String
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

    :param path: path of the vocabulary
    :type path: String

    :return: a dictionary with id as a key and word as value,
                 returns False if the vocabulary is not found
    :rtype: Dict
    """
    if Path(path).is_file():
        vocabulary_file = open(path,)
        return json.load(vocabulary_file)
    return False
