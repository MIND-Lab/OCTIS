from abc import ABC, abstractmethod
import os
import numpy as np
import json


class AbstractModel(ABC):
    """
    Class structure of a generic Topic Modeling implementation
    """

    def __init__(self):
        """
        Create a blank model to initialize
        """
        self.hyperparameters = dict()

    def set_hyperparameters(self, **kwargs):
        """
        Set model hyperparameters

        :param **kwargs: a dictionary of in the form {hyperparameter name: value}
        """
        for key, value in kwargs.items():
            self.hyperparameters[key] = value

    @abstractmethod
    def train_model(self, dataset, hyperparameters, top_words=10):
        """
        Train the model.
        :param dataset: Dataset
        :param hyperparameters: dictionary in the form {hyperparameter name: value}
        :param top_words: number of top significant words for each topic (default: 10)

        :return model_output: a dictionary containing up to 4 keys: *topics*, *topic-word-matrix*,
        *topic-document-matrix*, *test-topic-document-matrix*. *topics* is the list of the most significant words for
        each topic (list of lists of strings). *topic-word-matrix* is the matrix (num topics x ||vocabulary||)
        containing  the probabilities of a word in a given topic. *topic-document-matrix* is the matrix (||topics|| x
        ||training documents||) containing the probabilities of the topics in a given training document.
        *test-topic-document-matrix* is the matrix (||topics|| x ||testing documents||) containing the probabilities
        of the topics in a given testing document.
        """
        pass


def save_model_output(model_output, path=os.curdir, appr_order=7):
    """
    Saves the model output in the chosen directory

    :param model_output: output of the model
    :param path: path in which the file will be saved and name of the file
    :param appr_order: approximation order (used to round model_output values)
    """

    to_save = {}
    try:
        for single_output in model_output.keys():
            if single_output != "topics" and single_output != "test-topics":
                to_save[single_output] = (
                    model_output[single_output].round(appr_order))
            else:
                to_save[single_output] = (model_output[single_output])
        np.savez_compressed(path, **to_save)
    except:
        raise Exception("error in saving the output model file")


def load_model_output(output_path, vocabulary_path=None, top_words=10):
    """
    Loads a model output from the choosen directory

    Parameters
    ----------
    :param output_path: path in which th model output is saved
    :param vocabulary_path: path in which the vocabulary is saved (optional, used to retrieve the top k words of each
     topic)
    :param top_words: top k words to retrieve for each topic (in case a vocabulary path is given)
    """
    output = dict(np.load(output_path, allow_pickle=True))
    if vocabulary_path is not None:
        vocabulary_file = open(vocabulary_path, 'r')
        vocabulary = json.load(vocabulary_file)
        index2vocab = vocabulary

        topics_output = []
        for topic in output["topic-word-matrix"]:
            top_k = np.argsort(topic)[-top_words:]
            top_k_words = list(
                reversed([[index2vocab[str(i)], float(topic[i])] for i in top_k]))
            topics_output.append(top_k_words)

        output["topic-word-matrix"] = output["topic-word-matrix"].tolist()
        output["topic-document-matrix"] = output["topic-document-matrix"].tolist()
        if "test-topic-word-matrix" in output:
            output["test-topic-word-matrix"] = output["test-topic-word-matrix"].tolist()
        if "test-topic-document-matrix" in output:
            output["test-topic-document-matrix"] = output["test-topic-document-matrix"].tolist()

        output["topics"] = topics_output
    return output
