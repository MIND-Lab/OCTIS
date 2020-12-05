import os
import ast
import sys
import json
import inspect
import importlib
from pathlib import Path
from importlib import util

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


#OK IT IS WORKING, WOW I'M AMAZED
datasetClass = importDataset()
dataset = datasetClass()
dataset.load("optopic/preprocessed_datasets/m10_validation")

modelClass = importModel("LDA")
model = modelClass()

model.hyperparameters.update({"num_topics":10})
model.partitioning(False)


metric_parameters = {"topk":10}

metricClass = importMetric("Topic_diversity","diversity_metrics")
topicDiversity = metricClass(metric_parameters)

output = model.train_model(dataset)
print(topicDiversity.score(output))