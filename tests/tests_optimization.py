"""Tests for `octis` package related to Hyper-parameter optimization"""

import os
import json

import pytest

# %% load the libraries
from octis.models.LDA import LDA
from octis.dataset.dataset import Dataset
from octis.optimization.optimizer import Optimizer
from skopt.space.space import Real, Categorical
from octis.evaluation_metrics.coherence_metrics import Coherence

os.chdir(os.path.pardir)

@pytest.fixture
def root_dir():
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def data_dir(root_dir):
    return root_dir + "/../preprocessed_datasets/"

@pytest.fixture
def data_dir_test():
    return "tests_optimization/"

def test_simple_optimization(data_dir,data_dir_test):

    #Load dataset
    dataset = Dataset()
    dataset.load_custom_dataset(data_dir + '/M10')

    #Load model
    model = LDA(num_topics=5, iterations=200)

    #Choose of the metric function to optimize
    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi = Coherence(metric_parameters)

    #Create search space for optimization
    search_space = {
        "alpha": Real(low=0.001, high=5.0),
        "eta": Real(low=0.001, high=5.0)
    }

    save_path = data_dir_test+"test_simple_optimization/"

    #Choose number of call and number of model_runs
    number_of_call = 5
    model_runs = 3
    n_random_starts = 3

    #Optimize the function npmi using Bayesian Optimization (simple Optimization)
    optimizer = Optimizer()
    BestObject=optimizer.optimize(model, dataset, npmi, search_space,
                       number_of_call=number_of_call,
                       model_runs=model_runs,
                       n_random_starts = n_random_starts,
                       save_path=save_path)

    #check the integrity of BestObject
    assert BestObject.info["metric_name"] == npmi.__class__.__name__
    assert BestObject.info["surrogate_model"] == "RF"
    assert BestObject.info["acq_func"] == "LCB"
    assert BestObject.info["optimization_type"] == "Maximize"

    assert BestObject.info["model_runs"] == model_runs
    assert BestObject.info["number_of_call"] == number_of_call
    assert BestObject.info["n_random_starts"] == n_random_starts

    assert len(BestObject.func_vals) == number_of_call
    assert len(BestObject.info["x_iters"]) == 2
    assert all([len(BestObject.info["x_iters"][el]) == number_of_call for el in BestObject.info["x_iters"].keys()])
    assert all([len(BestObject.info["dict_model_runs"][npmi.__class__.__name__][el]) == model_runs for el in BestObject.info["dict_model_runs"][npmi.__class__.__name__].keys()])

    #check the integrity of the json file
    assert os.path.isfile(save_path + "result.json")

    for i in range(number_of_call):
        for j in range(model_runs):
            assert os.path.isfile(save_path+"models/"+str(i)+"_"+str(j)+".npz")

    f = open(save_path+"result.json")
    file=json.load(f)

    assert len(file) == 37
    assert file["metric_name"] == npmi.__class__.__name__
    assert file["surrogate_model"] == "RF"
    assert file["acq_func"] == "LCB"
    assert file["optimization_type"] == "Maximize"

    assert len(file["f_val"]) == number_of_call
    assert all([type(el) == float for el in file["f_val"]])

    assert len(file["x_iters"]) == 2
    assert all([len(file["x_iters"][el]) == number_of_call for el in file["x_iters"].keys()])

    assert len(file["dict_model_runs"]) == 1
    assert all([len(file["dict_model_runs"][npmi.__class__.__name__][el]) == model_runs for el in file["dict_model_runs"][npmi.__class__.__name__].keys()])

def test_resume_optimization(data_dir,data_dir_test):
    #Load dataset
    dataset = Dataset()
    dataset.load_custom_dataset(data_dir + '/M10')

    #Load model
    model = LDA(num_topics=5, iterations=200)

    #Choose of the metric function to optimize
    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi = Coherence(metric_parameters)

    #Create search space for optimization
    search_space = {
        "alpha": Real(low=0.001, high=5.0),
        "eta": Real(low=0.001, high=5.0)
    }

    #Choose number of call and number of model_runs
    number_of_call = 4
    model_runs = 3
    n_random_starts = 3

    save_path = data_dir_test+"test_resume_optimization/"

    #Optimize the function npmi using Bayesian Optimization (simple Optimization)
    optimizer = Optimizer()
    BestObject = optimizer.optimize(model, dataset, npmi, search_space,
                                    number_of_call=number_of_call,
                                    model_runs=model_runs,
                                    n_random_starts=n_random_starts,
                                    save_path=save_path)
    #Resume the optimization
    extra_evaluations=2
    path = BestObject.name_json
    optimizer = Optimizer()
    BestObject=optimizer.resume_optimization(path, extra_evaluations = extra_evaluations)

    #check the integrity of BestObject
    assert BestObject.info["metric_name"] == npmi.__class__.__name__
    assert BestObject.info["surrogate_model"] == "RF"
    assert BestObject.info["acq_func"] == "LCB"
    assert BestObject.info["optimization_type"] == "Maximize"

    assert BestObject.info["model_runs"] == model_runs
    assert BestObject.info["number_of_call"] == number_of_call+extra_evaluations
    assert BestObject.info["n_random_starts"] == n_random_starts

    assert len(BestObject.func_vals) == number_of_call+extra_evaluations
    assert len(BestObject.info["x_iters"]) == 2
    assert all([len(BestObject.info["x_iters"][el]) == number_of_call+extra_evaluations for el in BestObject.info["x_iters"].keys()])
    assert all([len(BestObject.info["dict_model_runs"][npmi.__class__.__name__][el]) == model_runs for el in BestObject.info["dict_model_runs"][npmi.__class__.__name__].keys()])

    #check the integrity of the json file
    assert os.path.isfile(save_path + "result.json")

    for i in range(number_of_call):
        for j in range(model_runs):
            assert os.path.isfile(save_path+"models/"+str(i)+"_"+str(j)+".npz")

    f = open(save_path+"result.json")
    file=json.load(f)

    assert len(file) == 37
    assert file["metric_name"] == npmi.__class__.__name__
    assert file["surrogate_model"] == "RF"
    assert file["acq_func"] == "LCB"
    assert file["optimization_type"] == "Maximize"

    assert len(file["f_val"]) == number_of_call+extra_evaluations
    assert all([type(el) == float for el in file["f_val"]])

    assert len(file["x_iters"]) == 2
    assert all([len(file["x_iters"][el]) == number_of_call+extra_evaluations for el in file["x_iters"].keys()])

    assert len(file["dict_model_runs"]) == 1
    assert all([len(file["dict_model_runs"][npmi.__class__.__name__][el]) == model_runs for el in file["dict_model_runs"][npmi.__class__.__name__].keys()])

# # %%
# def test_optimization_graphics(save_path):
#     # %% Load dataset
#     dataset = Dataset()
#     dataset.load_custom_dataset(data_dir + '/M10')
#
#     # %% Load model
#     model = LDA()
#
#     # %% Set model hyperparameters (not optimized by BO)
#     model.hyperparameters.update({"num_topics": 25, "iterations": 200})
#     model.partitioning(False)
#
#     # %% Choose of the metric function to optimize
#     metric_parameters = {
#         'texts': dataset.get_corpus(),
#         'topk': 10,
#         'measure': 'c_npmi'
#     }
#     npmi = Coherence(metric_parameters)
#
#     # %% Create search space for optimization
#     search_space = {
#         "alpha": Real(low=0.001, high=5.0),
#         "eta": Real(low=0.001, high=5.0)
#     }
#     # %% Optimize the function npmi using Bayesian Optimization
#     optimizer = Optimizer()
#     optimizer.optimize(model, dataset, npmi, search_space,
#                        number_of_call=number_of_call,
#                        model_runs=model_runs,
#                        save_path=save_path,
#                        plot_model=True,
#                        plot_best_seen=True)
#
#

def test_optimization_acq_function(data_dir,data_dir_test):

    acq_names = ['PI', 'EI', 'LCB']

    #Load dataset
    dataset = Dataset()
    dataset.load_custom_dataset(data_dir + '/M10')

    #Load model
    model = LDA(num_topics=5, iterations=200)

    #Choose of the metric function to optimize
    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi = Coherence(metric_parameters)

    #Create search space for optimization
    search_space = {
        "alpha": Real(low=0.001, high=5.0),
        "eta": Real(low=0.001, high=5.0)
    }

    #Choose number of call and number of model_runs
    number_of_call = 5
    model_runs = 1
    n_random_starts = 3

    #Optimize the function npmi using Bayesian Optimization
    save_path = data_dir_test + "test_acq_fun/"

    for acq_name in acq_names:
        save_path_specific = save_path + '/' + acq_name
        optimizer = Optimizer()
        BestObject=optimizer.optimize(model, dataset, npmi, search_space,
                           number_of_call=number_of_call,
                           model_runs=model_runs,
                           save_path=save_path_specific,
                           acq_func=acq_name)

        assert os.path.isfile(save_path_specific + "/result.json")

        #check the integrity of BestObject
        assert BestObject.info["acq_func"] == acq_name

        f = open(save_path_specific+ "/result.json")
        file=json.load(f)

        assert file["acq_func"] == acq_name

def test_optimization_surrogate_model(data_dir,data_dir_test):

    surrogate_models = ['RF', 'GP', 'ET']

    #Load dataset
    dataset = Dataset()
    dataset.load_custom_dataset(data_dir + '/M10')

    #Load model
    model = LDA(num_topics=5, iterations=200)

    #Choose of the metric function to optimize
    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi = Coherence(metric_parameters)

    #Create search space for optimization
    search_space = {
        "alpha": Real(low=0.001, high=5.0),
        "eta": Real(low=0.001, high=5.0)
    }

    #Choose number of call and number of model_runs
    number_of_call = 5
    model_runs = 1
    n_random_starts = 3

    #Optimize the function npmi using Bayesian Optimization
    save_path = data_dir_test + "test_surrogate_fun/"

    for surrogate_model in surrogate_models:
        save_path_specific = save_path + '/' + surrogate_model
        optimizer = Optimizer()
        BestObject=optimizer.optimize(model, dataset, npmi, search_space,
                           number_of_call=number_of_call,
                           model_runs=model_runs,
                           save_path=save_path_specific,
                           surrogate_model=surrogate_model)

        assert os.path.isfile(save_path_specific + "/result.json")

        #check the integrity of BestObject
        assert BestObject.info["surrogate_model"] == surrogate_model

        f = open(save_path_specific+"/result.json")
        file=json.load(f)

        assert file["surrogate_model"] == surrogate_model

#
# def test_early_stop(save_path):
#     # %% Load dataset
#     dataset = Dataset()
#     dataset.load_custom_dataset(data_dir + '/M10')
#
#     # %% Load model
#     model = LDA()
#
#     # %% Set model hyperparameters (not optimized by BO)
#     model.hyperparameters.update({"num_topics": 25, "iterations": 200})
#     model.partitioning(False)
#     # %% Choose of the metric function to optimize
#     metric_parameters = {
#         'texts': dataset.get_corpus(),
#         'topk': 10,
#         'measure': 'c_npmi'
#     }
#     npmi = Coherence(metric_parameters)
#
#     # %% Create search space for optimization
#     search_space = {
#         "alpha": Real(low=0.001, high=5.0),
#         "eta": Real(low=0.001, high=5.0)
#     }
#     # %% Optimize the function npmi using Bayesian Optimization
#     optimizer = Optimizer()
#     optimizer.optimize(model, dataset, npmi, search_space,
#                        number_of_call=number_of_call,
#                        model_runs=model_runs,
#                        save_path=save_path,
#                        early_stop=True,
#                        early_step=2)
#
#
# def test_initial_point_generator(save_path):
#     initial_point_generators = ['lhs', 'sobol', 'halton', 'hammersly', 'grid', 'random']
#     # %% Load dataset
#     dataset = Dataset()
#     dataset.load_custom_dataset(data_dir + '/M10')
#
#     # %% Load model
#     model = LDA()
#
#     # %% Set model hyperparameters (not optimized by BO)
#     model.hyperparameters.update({"num_topics": 25, "iterations": 200})
#     model.partitioning(False)
#
#     # %% Choose of the metric function to optimize
#     metric_parameters = {
#         'texts': dataset.get_corpus(),
#         'topk': 10,
#         'measure': 'c_npmi'
#     }
#     npmi = Coherence(metric_parameters)
#
#     # %% Create search space for optimization
#     search_space = {
#         "alpha": Real(low=0.001, high=5.0),
#         "eta": Real(low=0.001, high=5.0)
#     }
#     # %% Optimize the function npmi using Bayesian Optimization
#     for initial_point_generator in initial_point_generators:
#         save_path_specific = save_path + '/' + initial_point_generator
#         optimizer = Optimizer()
#         optimizer.optimize(model, dataset, npmi, search_space,
#                            number_of_call=10,
#                            model_runs=model_runs,
#                            save_path=save_path_specific,
#                            n_random_starts=5,
#                            initial_point_generator=initial_point_generator)
#
#
# # %%
# def test_extra_metrics(save_path):
#     # %% Load dataset
#     dataset = Dataset()
#     dataset.load_custom_dataset(data_dir + '/M10')
#
#     # %% Load model
#     model = LDA()
#
#     # %% Set model hyperparameters (not optimized by BO)
#     model.hyperparameters.update({"num_topics": 25, "iterations": 200})
#     model.partitioning(False)
#
#     # %% Choose of the metric function to optimize
#     metric_parameters = {
#         'texts': dataset.get_corpus(),
#         'topk': 10,
#         'measure': 'c_npmi'
#     }
#     npmi = Coherence(metric_parameters)
#
#     # %% Choose the extra metric to compute
#     metric_parameters = {
#         'texts': dataset.get_corpus(),
#         'topk': 10,
#         'measure': 'c_npmi'
#     }
#     npmi2 = Coherence(metric_parameters)
#     # %% Create search space for optimization
#     search_space = {
#         "alpha": Real(low=0.001, high=5.0),
#         "eta": Real(low=0.001, high=5.0)
#     }
#     # %% Optimize the function npmi using Bayesian Optimization (simple Optimization)
#     optimizer = Optimizer()
#     optimizer.optimize(model, dataset, npmi, search_space,
#                        number_of_call=number_of_call,
#                        model_runs=model_runs,
#                        save_path=save_path,
#                        extra_metrics=[npmi2],
#                        save_models=False,
#                        plot_best_seen=True,
#                        plot_model=True)
#
#
# def test_extra_metrics_resume(save_path):
#     # %% Load dataset
#     dataset = Dataset()
#     dataset.load_custom_dataset(data_dir + '/M10')
#
#     # %% Load model
#     model = LDA()
#
#     # %% Set model hyperparameters (not optimized by BO)
#     model.hyperparameters.update({"num_topics": 25, "iterations": 200})
#     model.partitioning(False)
#
#     # %% Choose of the metric function to optimize
#     metric_parameters = {
#         'texts': dataset.get_corpus(),
#         'topk': 10,
#         'measure': 'c_npmi'
#     }
#     npmi = Coherence(metric_parameters)
#
#     # %% Choose the extra metric to compute
#     metric_parameters = {
#         'texts': dataset.get_corpus(),
#         'topk': 10,
#         'measure': 'c_npmi'
#     }
#     npmi2 = Coherence(metric_parameters)
#     # %% Create search space for optimization
#     search_space = {
#         "alpha": Real(low=0.001, high=5.0),
#         "eta": Real(low=0.001, high=5.0)
#     }
#     # %% Optimize the function npmi using Bayesian Optimization (simple Optimization)
#     optimizer = Optimizer()
#     BestObject = optimizer.optimize(model, dataset, npmi, search_space,
#                                     number_of_call=number_of_call,
#                                     model_runs=model_runs,
#                                     save_path=save_path,
#                                     extra_metrics=[npmi2])
#     # %% Resume the optimization
#     path = BestObject.name_json
#     optimizer = Optimizer()
#     optimizer.resume_optimization(path, extra_evaluations=3)
#
#
# def test_extra_metrics_initial_input_points(save_path):
#     # %% Load dataset
#     dataset = Dataset()
#     dataset.load_custom_dataset(data_dir + '/M10')
#
#     # %% Load model
#     model = LDA()
#
#     # %% Set model hyperparameters (not optimized by BO)
#     model.hyperparameters.update({"num_topics": 25, "iterations": 200})
#     model.partitioning(False)
#
#     # %% Choose of the metric function to optimize
#     metric_parameters = {
#         'texts': dataset.get_corpus(),
#         'topk': 10,
#         'measure': 'c_npmi'
#     }
#     npmi = Coherence(metric_parameters)
#
#     # %% Create search space for optimization
#     search_space = {
#         "alpha": Real(low=0.001, high=5.0),
#         "eta": Real(low=0.001, high=5.0)
#     }
#
#     # %% inizialization of x0 as a dict
#     x0 = {"eta": [0.1, 0.5, 0.5], "alpha": [0.5, 0.1, 0.5]}
#
#     # %% Optimize the function npmi using Bayesian Optimization (simple Optimization)
#     optimizer = Optimizer()
#     optimizer.optimize(model, dataset, npmi, search_space,
#                        number_of_call=number_of_call,
#                        x0=x0,
#                        y0=[-0.1, 0.1, -0.1],
#                        plot_best_seen=True,
#                        plot_model=True,
#                        model_runs=model_runs,
#                        save_path=save_path + 'test1/')
#     # %% Optimize the function npmi using Bayesian Optimization (simple Optimization)
#     optimizer = Optimizer()
#     optimizer.optimize(model, dataset, npmi, search_space,
#                        number_of_call=number_of_call,
#                        x0=x0,
#                        plot_best_seen=True,
#                        plot_model=True,
#                        model_runs=model_runs,
#                        save_path=save_path + 'test2/')
#
#
