import os
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

    datasets = os.listdir(str(os.path.join(path,"preprocessed_datasets")))
    datasets.remove("README.txt")
    return datasets
