"""Tests for `octis` package related to Hyper-parameter optimization"""

import json
import os

import pytest
from skopt.space.space import Real

from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.classification_metrics import F1Score
# %% load the libraries
from octis.models.LDA import LDA
from octis.optimization.optimizer import Optimizer

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


@pytest.fixture
def dataset(data_dir):
    # Load dataset
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir + '/M10')

    return dataset


@pytest.fixture
def model():
    # Load model
    model = LDA(num_topics=5, iterations=200)

    return model


@pytest.fixture
def metric(dataset):
    # Choose of the metric function to optimize
    npmi = Coherence(texts=dataset.get_corpus(), topk=10, measure='c_npmi')
    return npmi


@pytest.fixture
def extra_metric(dataset):
    # extra metric
    f1_metric = F1Score(dataset=dataset)
    return f1_metric


@pytest.fixture
def search_space():
    # Create search space for optimization
    search_space = {
        "alpha": Real(low=0.001, high=5.0),
        "eta": Real(low=0.001, high=5.0)
    }

    return search_space


def test_simple_optimization(dataset, model, metric, search_space, data_dir_test):
    save_path = data_dir_test + "test_simple_optimization/"
    save_name = "result_simple_optimization.json"
    # Choose number of call and number of model_runs
    number_of_call = 5
    model_runs = 2
    n_random_starts = 3

    # Optimize the metric using Bayesian Optimization (simple Optimization)
    optimizer = Optimizer()
    optimization_result = optimizer.optimize(
        model, dataset, metric, search_space, number_of_call=number_of_call,
        model_runs=model_runs, n_random_starts=n_random_starts, save_path=save_path,
        save_name=save_name)

    # check the integrity of optimization_result
    assert optimization_result.info["metric_name"] == metric.__class__.__name__
    assert type(optimization_result.info["metric_name"]) == str
    assert optimization_result.info["surrogate_model"] == "RF"
    assert optimization_result.info["acq_func"] == "LCB"
    assert optimization_result.info["optimization_type"] == "Maximize"

    assert optimization_result.info["model_runs"] == model_runs
    assert optimization_result.info["number_of_call"] == number_of_call
    assert optimization_result.info["n_random_starts"] == n_random_starts

    assert len(optimization_result.func_vals) == number_of_call
    assert len(optimization_result.info["x_iters"]) == 2
    assert all([len(optimization_result.info["x_iters"][el]) == number_of_call for el in
                optimization_result.info["x_iters"].keys()])
    assert all([len(optimization_result.info["dict_model_runs"][metric.__class__.__name__][el]) == model_runs for el in
                optimization_result.info["dict_model_runs"][metric.__class__.__name__].keys()])

    # check the existence of the output file
    assert os.path.isfile(save_path + save_name)

    for i in range(number_of_call):
        for j in range(model_runs):
            assert os.path.isfile(save_path + "models/" + str(i) + "_" + str(j) + ".npz")

    # check the integrity of the json file
    f = open(save_path + save_name)
    file = json.load(f)

    assert len(file) == 38
    assert type(file["metric_name"]) == str
    assert file["metric_name"] == metric.__class__.__name__
    assert file["surrogate_model"] == "RF"
    assert file["acq_func"] == "LCB"
    assert file["optimization_type"] == "Maximize"

    assert file["initial_point_generator"] == "lhs"

    assert len(file["f_val"]) == number_of_call
    assert all([type(el) == float for el in file["f_val"]])

    assert len(file["x_iters"]) == 2
    assert all([len(file["x_iters"][el]) == number_of_call for el in file["x_iters"].keys()])

    assert len(file["dict_model_runs"]) == 1
    assert all([len(file["dict_model_runs"][metric.__class__.__name__][el]) == model_runs for el in
                file["dict_model_runs"][metric.__class__.__name__].keys()])


def test_resume_optimization(dataset, model, metric, search_space, data_dir_test):
    # Choose number of call and number of model_runs
    number_of_call = 4
    model_runs = 2
    n_random_starts = 3

    save_path = data_dir_test + "test_resume_optimization/"

    # Optimize the function npmi using Bayesian Optimization (simple Optimization)
    optimizer = Optimizer()
    optimization_result = optimizer.optimize(
        model, dataset, metric, search_space, number_of_call=number_of_call,
        model_runs=model_runs, n_random_starts=n_random_starts, save_path=save_path)

    # Resume the optimization
    extra_evaluations = 2
    path = optimization_result.name_json
    optimizer = Optimizer()
    optimization_result = optimizer.resume_optimization(path, extra_evaluations=extra_evaluations)

    # check the integrity of optimization_result
    assert optimization_result.info["number_of_call"] == number_of_call + extra_evaluations
    assert len(optimization_result.func_vals) == number_of_call + extra_evaluations
    assert all([len(optimization_result.info["x_iters"][el]) == number_of_call + extra_evaluations for el in
                optimization_result.info["x_iters"].keys()])

    # check the existence of the output file
    assert os.path.isfile(save_path + "result.json")

    for i in range(number_of_call + extra_evaluations):
        for j in range(model_runs):
            assert os.path.isfile(save_path + "models/" + str(i) + "_" + str(j) + ".npz")

    # check the integrity of the json file
    f = open(save_path + "result.json")
    file = json.load(f)

    assert len(file) == 38
    assert len(file["f_val"]) == number_of_call + extra_evaluations
    assert all([len(file["x_iters"][el]) == number_of_call + extra_evaluations for el in file["x_iters"].keys()])


def test_optimization_graphics(dataset, model, metric, search_space, data_dir_test):
    # Choose number of call and number of model_runs
    number_of_call = 4
    model_runs = 3
    n_random_starts = 3

    # Optimize the function npmi using Bayesian Optimization
    save_path = data_dir_test + "test_plot_fun/"

    # Optimize the function npmi using Bayesian Optimization
    optimizer = Optimizer()
    optimizer.optimize(model, dataset, metric, search_space,
                       number_of_call=number_of_call, model_runs=model_runs, save_path=save_path, plot_model=True,
                       plot_best_seen=True, plot_name="B0_plot")

    assert os.path.isfile(save_path + "result.json")
    assert os.path.isfile(save_path + "B0_plot_model_runs_Coherence.png")
    assert os.path.isfile(save_path + "B0_plot_best_seen.png")


def test_optimization_acq_function(dataset, model, metric, search_space, data_dir_test):
    acq_names = ['PI', 'EI', 'LCB']
    # Choose number of call and number of model_runs
    number_of_call = 4
    model_runs = 2
    n_random_starts = 3

    # Optimize the function npmi using Bayesian Optimization
    save_path = data_dir_test + "test_acq_fun/"

    for acq_name in acq_names:
        save_path_specific = save_path + '/' + acq_name
        optimizer = Optimizer()
        optimization_result = optimizer.optimize(
            model, dataset, metric, search_space, number_of_call=number_of_call,
            model_runs=model_runs, save_path=save_path_specific, acq_func=acq_name)

        assert os.path.isfile(save_path_specific + "/result.json")
        assert optimization_result.info["acq_func"] == acq_name

        f = open(save_path_specific + "/result.json")
        file = json.load(f)

        assert file["acq_func"] == acq_name


def test_optimization_surrogate_model(dataset, model, metric, search_space, data_dir_test):
    surrogate_models = ['RF', 'GP', 'ET']

    # Choose number of call and number of model_runs
    number_of_call = 4
    model_runs = 2
    n_random_starts = 3

    # Optimize the function npmi using Bayesian Optimization
    save_path = data_dir_test + "test_surrogate_fun/"

    for surrogate_model in surrogate_models:
        save_path_specific = save_path + '/' + surrogate_model
        optimizer = Optimizer()
        BestObject = optimizer.optimize(model, dataset, metric, search_space,
                                        number_of_call=number_of_call,
                                        model_runs=model_runs,
                                        n_random_starts=n_random_starts,
                                        save_path=save_path_specific,
                                        surrogate_model=surrogate_model)

        assert os.path.isfile(save_path_specific + "/result.json")
        assert BestObject.info["surrogate_model"] == surrogate_model

        f = open(save_path_specific + "/result.json")
        file = json.load(f)

        assert file["surrogate_model"] == surrogate_model


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

def test_extra_metrics(dataset, model, metric, extra_metric, search_space, data_dir_test):
    # Choose number of call and number of model_runs
    number_of_call = 4
    model_runs = 2
    n_random_starts = 3

    # Optimize the function npmi using Bayesian Optimization
    save_path = data_dir_test + "test_extrametrics_fun/"

    # %% Optimize the function npmi using Bayesian Optimization (simple Optimization)
    optimizer = Optimizer()
    optimizer.optimize(model, dataset, metric, search_space,
                       number_of_call=number_of_call,
                       model_runs=model_runs,
                       n_random_starts=n_random_starts,
                       save_path=save_path,
                       extra_metrics=[extra_metric],
                       plot_model=True)

    assert os.path.isfile(save_path + "result.json")
    assert os.path.isfile(save_path + "B0_plot_model_runs_Coherence.png")
    assert os.path.isfile(save_path + "B0_plot_model_runs_0_F1Score.png")
    assert not os.path.isfile(save_path + "B0_plot_best_seen.png")

    f = open(save_path + "result.json")
    file = json.load(f)

    assert len(file) == 38
    assert file["metric_name"] == metric.__class__.__name__
    assert len(file["extra_metric_names"]) == 1
    assert file["extra_metric_names"][0] == "0_F1Score"

    assert len(file["dict_model_runs"]) == 2

    assert all([len(file["dict_model_runs"][metric.__class__.__name__][el]) == model_runs for el in
                file["dict_model_runs"][metric.__class__.__name__].keys()])

    assert all([len(file["dict_model_runs"]["0_" + extra_metric.__class__.__name__][el]) == model_runs for el in
                file["dict_model_runs"]["0_" + extra_metric.__class__.__name__].keys()])


def test_extra_metrics_resume(dataset, model, metric, extra_metric, search_space, data_dir_test):
    # Choose number of call and number of model_runs
    number_of_call = 4
    model_runs = 2
    n_random_starts = 3

    # Optimize the function npmi using Bayesian Optimization
    save_path = data_dir_test + "test_extrametricsResume_fun/"

    # Optimize the function npmi using Bayesian Optimization (simple Optimization)
    optimizer = Optimizer()
    optimization_result = optimizer.optimize(model, dataset, metric, search_space,
                                             number_of_call=number_of_call,
                                             model_runs=model_runs,
                                             n_random_starts=n_random_starts,
                                             save_path=save_path,
                                             extra_metrics=[extra_metric],
                                             plot_best_seen=True,
                                             plot_model=True)

    # # Resume the optimization
    extra_evaluations = 2
    path = optimization_result.name_json
    optimizer = Optimizer()
    optimization_result = optimizer.resume_optimization(path, extra_evaluations=extra_evaluations)

    # check the integrity of the json file
    assert os.path.isfile(save_path + "result.json")
    assert os.path.isfile(save_path + "B0_plot_model_runs_Coherence.png")
    assert os.path.isfile(save_path + "B0_plot_model_runs_0_F1Score.png")
    assert os.path.isfile(save_path + "B0_plot_best_seen.png")

    f = open(save_path + "result.json")
    file = json.load(f)

    assert len(file) == 38
    assert len(file["f_val"]) == number_of_call + extra_evaluations
    assert all([len(file["x_iters"][el]) == number_of_call + extra_evaluations for el in file["x_iters"].keys()])

    assert len(file["dict_model_runs"]) == 2
    assert all([len(file["dict_model_runs"][metric.__class__.__name__][el]) == model_runs for el in
                file["dict_model_runs"][metric.__class__.__name__].keys()])
    assert all([len(file["dict_model_runs"]["0_" + extra_metric.__class__.__name__][el]) == model_runs for el in
                file["dict_model_runs"]["0_" + extra_metric.__class__.__name__].keys()])


def test_initial_input(dataset, model, metric, extra_metric, search_space, data_dir_test):
    # Choose number of call and number of model_runs
    number_of_call = 4
    model_runs = 2

    # inizialization of x0 as a dict
    x0 = {"eta": [0.1, 0.5, 0.5], "alpha": [0.5, 0.1, 0.5]}

    # Optimize the function npmi using Bayesian Optimization
    save_path = data_dir_test + "test_initial_output_fun/"

    # Optimize the function npmi using Bayesian Optimization (simple Optimization)
    optimizer = Optimizer()
    optimization_result = optimizer.optimize(model, dataset, metric, search_space,
                                             number_of_call=number_of_call,
                                             model_runs=model_runs,
                                             x0=x0,
                                             save_path=save_path,
                                             extra_metrics=[metric])

    assert os.path.isfile(save_path + "result.json")

    f = open(save_path + "result.json")
    file = json.load(f)

    assert len(file) == 38
    assert len(file["x0"]["eta"]) == 3
    assert type(file["x0"]["eta"]) == list
    assert all([file["x0"]["eta"][i] == x0["eta"][i] for i in range(len(x0["eta"]))])

    assert len(file["x0"]["alpha"]) == 3
    assert type(file["x0"]["alpha"]) == list
    assert all([file["x0"]["alpha"][i] == x0["alpha"][i] for i in range(len(x0["alpha"]))])
