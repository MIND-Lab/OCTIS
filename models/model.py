from abc import ABC, abstractmethod
import os
import json
import numpy as np


class Abstract_Model(ABC):
    """
    Class structure of a generic Topic Modelling implementation
    """

    hyperparameters = {}

    def __init__(self):
        """
        Create a blank model to initialize
        """

    @abstractmethod
    def train_model(self, dataset, hyperparameters):
        """
        Train the model.
        Return a dictionary with up to 3 entries,
        'topics', 'topic-word-matrix' and 'topic-document-matrix'.
        'topics' is the list of the most significative words for
        each topic (list of lists of strings).
        'topic-word-matrix' is an NxV matrix of weights where N is the number
        of topics and V is the vocabulary length.
        'topic-document-matrix' is an NxD matrix of weights where N is the number
        of topics and D is the number of documents in the corpus.

        """
        pass

def save_model_output(model_output, path=os.curdir, appr_order=7, topics=True):
    """
    Saves the model output in the choosen directory

    Parameters
    ----------
    model_output: output of the model
    path: path in which the file will be saved
    appr_order: approximation order (used to round model_output values)
    topics: Boolean flag, default True. 
            If True the most important words for each topic
            will be saved.
    """

    if topics and ("topics" in model_output):
        file = open("topics.json", "w")
        json.dump(model_output["topics"], file)
        file.close()

    if topics and ("test-topics" in model_output):
        file = open("test-topics.json", "w")
        json.dump(model_output["test-topics"], file)
        file.close()

    to_save = {}
    for single_output in model_output.keys():
        if single_output != "topics" and single_output != "test-topics":
            to_save[single_output] = (
                model_output[single_output].round(appr_order))
    np.savez_compressed(
        "model_output",
        **to_save)