import os

# get the path to the framework folder
path = str(os.path.dirname(os.path.realpath(__file__)))
path = path[:path.rfind("/")]

def scanDatasets():
    """
    Retrieves the name of each dataset present in the framework

    Returns
    -------
    res : list with name of each dataset as element
    """

    datasets = os.listdir(path+"/preprocessed_datasets")
    datasets.remove("README.txt")
    return datasets