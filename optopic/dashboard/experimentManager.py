import os
import ast
import sys
import json
import numpy as np
import inspect
import importlib
from pathlib import Path
from importlib import util
import gensim.corpora as corpora
import optopic.configuration.defaults as defaults
from skopt.space.space import Real, Categorical, Integer
from optopic.models.model import load_model_output


path = Path(os.path.dirname(os.path.realpath(__file__)))
path = str(path.parent)

# Import optopic module
spec = importlib.util.spec_from_file_location(
    "optopic", path+"/__init__.py", submodule_search_locations=[])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
importlib.invalidate_caches()


def importClass(className, moduleName, modulePath):
    """
    Import a class runtime based on its module and name

    Parameters
    ----------
    className : name of the class
    moduleName : name of the module
    modulePath: absolute path to the module

    Returns
    singleClass : returns the selected class
    """
    spec = importlib.util.spec_from_file_location(
        moduleName, modulePath, submodule_search_locations=[])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    importlib.invalidate_caches()
    singleClass = getattr(module, className)
    return singleClass


def importModel(modelName):
    modulePath = os.path.join(path, "models")
    modulePath = os.path.join(modulePath, modelName+".py")
    model = importClass(modelName, modelName, modulePath)
    return model


def importMetric(metricName):
    modulePath = os.path.join(path, "evaluation_metrics")
    moduleName = defaults.metric_parameters[metricName]["module"]
    modulePath = os.path.join(modulePath, moduleName+".py")
    metric = importClass(metricName, metricName, modulePath)
    return metric


def importDataset():
    modulePath = os.path.join(path, "dataset")
    modulePath = os.path.join(modulePath, "dataset.py")
    datasetClass = importClass("Dataset", "dataset", modulePath)
    return datasetClass


def importOptimizer():
    modulePath = os.path.join(path, "optimization")
    modulePath = os.path.join(modulePath, "optimizer.py")
    optimizerClass = importClass("Optimizer", "optimizer", modulePath)
    return optimizerClass


# TO UPDATE
def startExperiment(parameters):
    """
    Starts an experiment with the given parameters
    """
    # Import dataset class and initialize an instance with the choosen dataset
    datasetClass = importDataset()
    dataset = datasetClass()
    datasetPath = path+"/preprocessed_datasets/"+parameters["dataset"]
    dataset.load(datasetPath)

    modelClass = importModel(parameters["model"]["name"])
    model = modelClass()

    model.hyperparameters.update(parameters["model"]["parameters"])
    model.partitioning(False)

    search_space = {}

    for key, value in parameters["optimization"]["search_spaces"].items():
        if "low" in value:
            if isinstance(value["low"], float) or isinstance(value["high"], float):
                search_space[key] = Real(low=value["low"], high=value["high"])
            else:
                search_space[key] = Integer(
                    low=value["low"], high=value["high"])
        else:
            search_space[key]: Categorical(value)

    metric_parameters = parameters["optimize_metrics"][0]["parameters"]
    for key in metric_parameters:
        if metric_parameters[key] == "use dataset texts":
            metric_parameters[key] = dataset.get_corpus()
        if os.path.isdir(str(metric_parameters[key])):
            metricDataset = datasetClass()
            metricDataset.load(metric_parameters[key])
            metric_parameters[key] = metricDataset.get_corpus()

    metricClass = importMetric(parameters["optimize_metrics"][0]["name"])
    metric = metricClass(metric_parameters)

    metrics_to_track = []
    for single_metric in parameters["track_metrics"]:
        metricClass = importMetric(single_metric["name"])
        single_metric_parameters = single_metric["parameters"]
        for key in single_metric_parameters:
            if single_metric_parameters[key] == "use dataset texts":
                single_metric_parameters[key] = dataset.get_corpus()
        new_metric = metricClass(single_metric_parameters)
        metrics_to_track.append(new_metric)

    vocabularyPath = str(os.path.join(
        parameters["path"], parameters["experimentId"], "models"))

    Path(vocabularyPath).mkdir(parents=True, exist_ok=True)

    vocabularyPath = str(os.path.join(vocabularyPath, "vocabulary.json"))

    file = open(vocabularyPath, "w")
    json.dump(dict(corpora.Dictionary(dataset.get_corpus())), file)
    file.close()

    Optimizer = importOptimizer()
    optimizer = Optimizer(model,
                          dataset,
                          metric,
                          search_space,
                          metrics_to_track,
                          random_state=True,
                          initial_point_generator="random",
                          surrogate_model=parameters["optimization"]["surrogate_model"],
                          model_runs=parameters["optimization"]["model_runs"],
                          n_random_starts=parameters["optimization"]["n_random_starts"],
                          acq_func=parameters["optimization"]["acquisition_function"],
                          number_of_call=parameters["optimization"]["iterations"],
                          save_models=True,
                          save_name=parameters["experimentId"],
                          save_path=str(os.path.join(parameters["path"], parameters["experimentId"])))

    optimizer.optimize()


def retrieveBoResults(path):
    """
    Function to load the results of BO
    Parameters
    ----------
    path : path where the results are saved (json file).
    Returns
    -------
    dict_return :dictionary 
    """
    # open json file
    with open(path, 'rb') as file:
        result = json.load(file)
    f_val = result['f_val']
    # output dictionart
    dict_return = dict()
    dict_return.update({"f_val": f_val})
    return dict_return


def retrieveIterationBoResults(path, iteration):
    """
    Function to load the results of BO until iteration
    Parameters
    ----------
    path : path where the results are saved (json file).
    iteration : considered iteration.
    Returns
    -------
    dict_return :dictionary 
    """
    # open json file
    path = "results/simple_GP/resultsBO.json"
    with open(path, 'rb') as file:
        result = json.load(file)
    values = result['f_val'][0:iteration]
    type_of_problem = result['optimization_type']
    if type_of_problem == 'Maximize':
        best_seen = max(values)
        worse_seen = min(values)
    else:
        best_seen = min(values)
        worse_seen = max(values)
    median_seen = np.median(values)
    mean_seen = np.mean(values)
    hyperparameters = result['x_iters']
    name_hyp = list(hyperparameters.keys())
    hyperparameters_iter = list()
    for name in name_hyp:
        hyperparameters_iter.append(hyperparameters[name][iteration])
    # dizionaro di output
    dict_return = dict()
    dict_return.update({"best_seen": best_seen})
    dict_return.update({"worse_seen": worse_seen})
    dict_return.update({"median_seen": median_seen})
    dict_return.update({"mean_seen": mean_seen})
    dict_return.update({"hyperparameter_names": name_hyp})
    dict_return.update({"hyperparameter_configuration": hyperparameters_iter})
    return dict_return

# Manca retrieve di Iperparametri iteerazione e topic-word-matrix e document-topic matrix


def singleInfo(path):
    """
    Metodo per calcolare media, mediana, best e worse della valutazioni delle funzioni obiettivo
    """
    with open(path, 'rb') as file:
        result = json.load(file)
    values = result['f_val']
    best_seen = max(values)
    worse_seen = min(values)
    median_seen = np.median(values)
    mean_seen = np.mean(values)
    # dizionaro di output
    dict_return = dict()
    dict_return.update({"best_seen": best_seen})
    dict_return.update({"worse_seen": worse_seen})
    dict_return.update({"median_seen": median_seen})
    dict_return.update({"mean_seen": mean_seen})

    return dict_return


def getModelInfo(path, iteration, modelRun):
    output = load_model_output(
        path+"/models/"+str(iteration)+"_"+str(modelRun)+".npz", path+"/models/vocabulary.json", 20)
    return output
