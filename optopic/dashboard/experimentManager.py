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
    "optopic", str(os.path.join(path, "__init__.py")), submodule_search_locations=[])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
importlib.invalidate_caches()


def importClass(class_name, module_name, module_path):
    """
    Import a class runtime based on its module and name

    Parameters
    ----------
    class_name : name of the class
    module_name : name of the module
    module_path: absolute path to the module

    Returns
    single_class : returns the selected class
    """
    spec = importlib.util.spec_from_file_location(
        module_name, module_path, submodule_search_locations=[])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    importlib.invalidate_caches()
    single_class = getattr(module, class_name)
    return single_class


def importModel(model_name):
    module_path = os.path.join(path, "models")
    module_path = os.path.join(module_path, model_name + ".py")
    model = importClass(model_name, model_name, module_path)
    return model


def importMetric(metric_name):
    module_path = os.path.join(path, "evaluation_metrics")
    module_name = defaults.metric_parameters[metric_name]["module"]
    module_path = os.path.join(module_path, module_name + ".py")
    metric = importClass(metric_name, metric_name, module_path)
    return metric


def importDataset():
    module_path = os.path.join(path, "dataset")
    module_path = os.path.join(module_path, "dataset.py")
    dataset_class = importClass("Dataset", "dataset", module_path)
    return dataset_class


def importOptimizer():
    module_path = os.path.join(path, "optimization")
    module_path = os.path.join(module_path, "optimizer.py")
    optimizer_class = importClass("Optimizer", "optimizer", module_path)
    return optimizer_class


# TO UPDATE
def startExperiment(parameters):
    """
    Starts an experiment with the given parameters
    """

    optimizationPath = str(os.path.join(
        parameters["path"], parameters["experimentId"]))
    json_file = str(os.path.join(optimizationPath,
                                parameters["experimentId"] + ".json"))
    if os.path.isfile(json_file):
        Optimizer = importOptimizer()
        optimizer = Optimizer()
        optimizer.resume_optimization(json_file)
    else:
        # Import dataset class and initialize an instance with the choosen dataset
        dataset_class = importDataset()
        dataset = dataset_class()
        dataset_path = str(os.path.join(
            path, "preprocessed_datasets", parameters["dataset"]))
        dataset.load(dataset_path)

        model_class = importModel(parameters["model"]["name"])
        model = model_class()

        model.hyperparameters.update(parameters["model"]["parameters"])
        model.partitioning(parameters["partitioning"])

        search_space = {}

        for key, value in parameters["optimization"]["search_spaces"].items():
            if "low" in value:
                if isinstance(value["low"], float) or isinstance(value["high"], float):
                    search_space[key] = Real(
                        low=value["low"], high=value["high"])
                else:
                    search_space[key] = Integer(
                        low=value["low"], high=value["high"])
            else:
                search_space[key]: Categorical(value)

        metric_parameters = parameters["optimize_metrics"][0]["parameters"]
        for key in metric_parameters:
            if metric_parameters[key] == "use dataset texts":
                metric_parameters[key] = dataset.get_corpus()
            elif os.path.isdir(str(metric_parameters[key])):
                metricDataset = dataset_class()
                metricDataset.load(metric_parameters[key])
                metric_parameters[key] = metricDataset.get_corpus()

        metric_class = importMetric(parameters["optimize_metrics"][0]["name"])
        metric = metric_class(metric_parameters)

        metrics_to_track = []
        for single_metric in parameters["track_metrics"]:
            metric_class = importMetric(single_metric["name"])
            single_metric_parameters = single_metric["parameters"]
            for key in single_metric_parameters:
                if single_metric_parameters[key] == "use dataset texts":
                    single_metric_parameters[key] = dataset.get_corpus()
            new_metric = metric_class(single_metric_parameters)
            metrics_to_track.append(new_metric)

        vocabulary_path = str(os.path.join(
            parameters["path"], parameters["experimentId"], "models"))

        Path(vocabulary_path).mkdir(parents=True, exist_ok=True)

        vocabulary_path = str(os.path.join(vocabulary_path, "vocabulary.json"))

        file = open(vocabulary_path, "w")
        json.dump(dict(corpora.Dictionary(dataset.get_corpus())), file)
        file.close()

        Optimizer = importOptimizer()
        optimizer = Optimizer()
        optimizer.optimize(model, dataset, metric, search_space, metrics_to_track, random_state=True,
                           initial_point_generator="random",
                           surrogate_model=parameters["optimization"]["surrogate_model"],
                           model_runs=parameters["optimization"]["model_runs"],
                           n_random_starts=parameters["optimization"]["n_random_starts"],
                           acq_func=parameters["optimization"]["acquisition_function"],
                           number_of_call=parameters["optimization"]["iterations"],
                           save_models=True, save_name=parameters["experimentId"], save_path=optimizationPath)


def retrieveBoResults(result_path):
    """
    Function to load the results_old of BO
    Parameters
    ----------
    result_path : path where the results_old are saved (json file).
    Returns
    -------
    dict_return :dictionary
    """
    if os.path.isfile(result_path):
        # open json file
        with open(result_path, 'rb') as file:
            result = json.load(file)
        f_val = result['f_val']
        # output dictionary
        dict_return = dict()
        dict_return.update({"f_val": f_val,
                            "current_iteration": result["current_call"],
                            "total_iterations": result["number_of_call"]})
        return dict_return
    return False

def retrieveIterationBoResults(path, iteration):
    """
    Function to load the results_old of BO until iteration
    Parameters
    ----------
    path : path where the results_old are saved (json file).
    iteration : considered iteration.
    Returns
    -------
    dict_return :dictionary
    """
    if os.path.isfile(path):
        # open json file
        with open(path, 'rb') as file:
            result = json.load(file)
        values = result["f_val"]

        print(values)
        type_of_problem = result['optimization_type']
        hyperparameters = result['x_iters']
        name_hyp = list(hyperparameters.keys())
        hyperparameters_iter = list()
        for name in name_hyp:
            hyperparameters_iter.append(hyperparameters[name][iteration])

        hyperparameters_config = dict(zip(name_hyp, hyperparameters_iter))
        # dizionaro di output
        dict_return = dict()

        metric_name = result["metric_name"]
        values=result['dict_model_runs'][metric_name]['iteration_' + str(iteration)]

        extra_metric_names=result["extra_metric_names"]
        for name in extra_metric_names:
            values = result['dict_model_runs'][name]['iteration_' + str(iteration)]
            dict_return.update({name+"_values": values})

        dict_return.update({"optimized_metric": result["metric_name"]})
        dict_return.update({"optimized_metric_values": values})
        dict_model_attributes=result['model_attributes']
        dict_return.update({"model_attributes": dict_model_attributes})
        dict_return.update({"model_name": result["model_name"]})
        dict_return.update(
            {"hyperparameter_configuration": hyperparameters_config})
        return dict_return
    return False


# Manca retrieve di Iperparametri iterazione e topic-word-matrix e document-topic matrix


def singleInfo(path):
    """
    Compute average, median, best and worst result of the object function evaluations
    """
    if os.path.isfile(path):
        with open(path, 'rb') as file:
            result = json.load(file)
        values = result['f_val']
        type_of_problem = result['optimization_type']
        if type_of_problem == 'Maximize':
            best_seen = max(values)
            worse_seen = min(values)
            best_index = np.argmax(values)
        else:
            best_seen = min(values)
            worse_seen = max(values)
            best_index = np.argmin(values)
        median_seen = np.median(values)
        mean_seen = np.mean(values)

        dict_metrics = result['dict_model_runs']
        model_runs=result['model_runs']
        name_metrics = list(dict_metrics.keys())
        dict_results = dict()
        for name in name_metrics:
            dict_results[name] = list()
            metric_results = dict_metrics[name]
            iterations = len(metric_results.keys())
            for i in range(iterations):
                dict_results[name].append(metric_results['iteration_' + str(i)])

        hyperparameters = result['x_iters']
        name_hyp = list(hyperparameters.keys())
        best_hyperparameter_configuration = dict()
        for name in name_hyp:
            best_hyperparameter_configuration.update(
                {name: hyperparameters[name][best_index]})
        # dizionaro di output
        dict_return = dict()
        dict_return.update({"model_runs": dict_results})
        dict_return.update({"f_val": values})
        dict_return.update({"best_seen": best_seen})
        dict_return.update({"worse_seen": worse_seen})
        dict_return.update({"median_seen": median_seen})
        dict_return.update({"mean_seen": mean_seen})
        dict_return.update({"model_runs": model_runs})
        dict_return.update({"current_iteration": result["current_call"],
                            "total_iterations": result["number_of_call"]})
        dict_return.update({"hyperparameter_configurations": hyperparameters})
        dict_return.update(
            {"hyperparameter_configuration": best_hyperparameter_configuration})
        dict_return.update({"optimized_metric": result["metric_name"]})

        #other hyper-parameter values
        dict_model_attributes=result['model_attributes']
        dict_return.update({"model_attributes": dict_model_attributes})
        dict_return.update({"model_name": result["model_name"]})

        dict_values_extra_metrics = dict()
        dict_stats_extra_metrics = dict()
        if len(result['extra_metric_names']) > 0:
            # nome delle metriche
            dict_return.update({"metric_names": name_metrics[1:]})
            extra_metrics_names = list(result['dict_model_runs'].keys())

            for name in extra_metrics_names[1:]:
                values = []
                dict_values = result['dict_model_runs'][name]
                iterations = list(dict_values.keys())
                for iter in iterations:
                    values.append(np.median(dict_values[iter]))
                dict_values_extra_metrics.update({name: values})
                val_stats = [np.max(values), np.min(
                    values), np.median(values), np.mean(values)]
                dict_stats_extra_metrics.update({name: val_stats})
            dict_return.update(
                {"extra_metric_vals": dict_values_extra_metrics})
            dict_return.update(
                {"extra_metric_stats": dict_stats_extra_metrics})
        else:
            dict_return.update({"metric_names": 0})
            dict_return.update({"extra_metric_vals": dict()})
            dict_return.update({"extra_metric_stats": dict()})

        return dict_return
    return False


def getModelInfo(experiment_path, iteration, modelRun):
    """
    Retrieve the output of the given model

    Parameters
    ----------
    experiment_path :  path of the experiment folder
    iteration : number of iteration
    modelRun : number of model run

    Returns
    -------
    output of the model and vocabulary
    """
    outputfile = str(os.path.join(experiment_path,
                                  "models",
                                  str(iteration) + "_" + str(modelRun) + ".npz"))
    vocabularyfile = str(os.path.join(experiment_path,
                                      "models",
                                      "vocabulary.json"))
    if os.path.isfile(outputfile) and os.path.isfile(vocabularyfile):
        output = load_model_output(outputfile, vocabularyfile, 20)
        return output
    return False
