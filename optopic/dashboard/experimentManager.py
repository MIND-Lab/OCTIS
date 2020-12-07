import os
import ast
import sys
import json
import inspect
import importlib
from pathlib import Path
from importlib import util
import optopic.configuration.defaults as defaults
from skopt.space.space import Real, Categorical, Integer


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


def importMetric(metricName, moduleName):
    modulePath = os.path.join(path, "evaluation_metrics")
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

    print(search_space)

    metric_parameters = parameters["optimize_metrics"][0]["parameters"]

    metricClass = importMetric(
        parameters["optimize_metrics"][0]["name"],
        defaults.metric_parameters[parameters["optimize_metrics"][0]["name"]]["module"])
    metric = metricClass(metric_parameters)

    Optimizer = importOptimizer()
    optimizer = Optimizer(model,
                          dataset,
                          metric,
                          search_space,
                          random_state=True,
                          initial_point_generator="random",
                          surrogate_model=parameters["optimization"]["surrogate_model"],
                          model_runs=parameters["optimization"]["model_runs"],
                          acq_func=parameters["optimization"]["acquisition_function"],
                          number_of_call=parameters["optimization"]["iterations"],
                          save_csv=True,
                          save_name=parameters["experimentId"],
                          save_path=parameters["path"])

    optimizer.optimize()
