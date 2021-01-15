"""Tests for `octis` package related to Hyper-parameter optimization"""

import os


# %% load the libraries
from octis.models.LDA import LDA
from octis.dataset.dataset import Dataset
from octis.optimization.optimizer import Optimizer
from skopt.space.space import Real, Categorical
from octis.evaluation_metrics.coherence_metrics import Coherence

os.chdir(os.path.pardir)

number_of_call = 5
model_runs = 3


# %%
def test_simple_optimization(save_path):
    # %% Load dataset
    dataset = Dataset()
    dataset.load("octis/preprocessed_datasets/M10")

    # %% Load model
    model = LDA()

    # %% Set model hyperparameters (not optimized by BO)
    model.hyperparameters.update({"num_topics": 25, "iterations": 200})
    model.partitioning(False)

    # %% Choose of the metric function to optimize
    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi = Coherence(metric_parameters)

    # %% Create search space for optimization
    search_space = {
        "alpha": Real(low=0.001, high=5.0),
        "eta": Real(low=0.001, high=5.0)
    }
    # %% Optimize the function npmi using Bayesian Optimization (simple Optimization)
    optimizer = Optimizer()
    optimizer.optimize(model, dataset, npmi, search_space,
                       number_of_call=number_of_call,
                       model_runs=model_runs,
                       save_path=save_path)


# %%
def test_resume_optimization(save_path):
    # %% Load dataset
    dataset = Dataset()
    dataset.load("octis/preprocessed_datasets/M10")

    # %% Load model
    model = LDA()

    # %% Set model hyperparameters (not optimized by BO)
    model.hyperparameters.update({"num_topics": 25, "iterations": 200})
    model.partitioning(False)

    # %% Choose of the metric function to optimize
    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi = Coherence(metric_parameters)

    # %% Create search space for optimization
    search_space = {
        "alpha": Real(low=0.001, high=5.0),
        "eta": Real(low=0.001, high=5.0)
    }
    # %% Optimize the function npmi using Bayesian Optimization (simple Optimization)
    optimizer = Optimizer()
    BestObject = optimizer.optimize(model, dataset, npmi, search_space,
                                    number_of_call=number_of_call,
                                    model_runs=model_runs,
                                    save_path=save_path)
    # %% Resume the optimization
    path = BestObject.name_json
    optimizer = Optimizer()
    optimizer.resume_optimization(path, extra_evaluations=3)


# %%
def test_optimization_graphics(save_path):
    # %% Load dataset
    dataset = Dataset()
    dataset.load("octis/preprocessed_datasets/M10")

    # %% Load model
    model = LDA()

    # %% Set model hyperparameters (not optimized by BO)
    model.hyperparameters.update({"num_topics": 25, "iterations": 200})
    model.partitioning(False)

    # %% Choose of the metric function to optimize
    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi = Coherence(metric_parameters)

    # %% Create search space for optimization
    search_space = {
        "alpha": Real(low=0.001, high=5.0),
        "eta": Real(low=0.001, high=5.0)
    }
    # %% Optimize the function npmi using Bayesian Optimization
    optimizer = Optimizer()
    optimizer.optimize(model, dataset, npmi, search_space,
                       number_of_call=number_of_call,
                       model_runs=model_runs,
                       save_path=save_path,
                       plot_model=True,
                       plot_best_seen=True)


def test_optimization_acq_function(save_path):
    acq_names = ['PI', 'EI', 'LCB']
    # %% Load dataset
    dataset = Dataset()
    dataset.load("octis/preprocessed_datasets/M10")

    # %% Load model
    model = LDA()

    # %% Set model hyperparameters (not optimized by BO)
    model.hyperparameters.update({"num_topics": 25, "iterations": 200})
    model.partitioning(False)

    # %% Choose of the metric function to optimize
    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi = Coherence(metric_parameters)

    # %% Create search space for optimization
    search_space = {
        "alpha": Real(low=0.001, high=5.0),
        "eta": Real(low=0.001, high=5.0)
    }
    # %% Optimize the function npmi using Bayesian Optimization
    for acq_name in acq_names:
        save_path_specific = save_path + '/' + acq_name
        optimizer = Optimizer()
        optimizer.optimize(model, dataset, npmi, search_space,
                           number_of_call=number_of_call,
                           model_runs=model_runs,
                           save_path=save_path_specific,
                           acq_func=acq_name)


def test_optimization_surrogate_model(save_path):
    surrogate_models = ['RF', 'RS', 'GP', 'ET']
    # %% Load dataset
    dataset = Dataset()
    dataset.load("octis/preprocessed_datasets/M10")

    # %% Load model
    model = LDA()

    # %% Set model hyperparameters (not optimized by BO)
    model.hyperparameters.update({"num_topics": 25, "iterations": 200})
    model.partitioning(False)

    # %% Choose of the metric function to optimize
    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi = Coherence(metric_parameters)

    # %% Create search space for optimization
    search_space = {
        "alpha": Real(low=0.001, high=5.0),
        "eta": Real(low=0.001, high=5.0)
    }
    # %% Optimize the function npmi using Bayesian Optimization
    for surrogate_model in surrogate_models:
        save_path_specific = save_path + '/' + surrogate_model
        optimizer = Optimizer()
        optimizer.optimize(model, dataset, npmi, search_space,
                           number_of_call=number_of_call,
                           model_runs=model_runs,
                           save_path=save_path_specific,
                           surrogate_model=surrogate_model)


def test_early_stop(save_path):
    # %% Load dataset
    dataset = Dataset()
    dataset.load("octis/preprocessed_datasets/M10")

    # %% Load model
    model = LDA()

    # %% Set model hyperparameters (not optimized by BO)
    model.hyperparameters.update({"num_topics": 25, "iterations": 200})
    model.partitioning(False)
    # %% Choose of the metric function to optimize
    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi = Coherence(metric_parameters)

    # %% Create search space for optimization
    search_space = {
        "alpha": Real(low=0.001, high=5.0),
        "eta": Real(low=0.001, high=5.0)
    }
    # %% Optimize the function npmi using Bayesian Optimization
    optimizer = Optimizer()
    optimizer.optimize(model, dataset, npmi, search_space,
                       number_of_call=number_of_call,
                       model_runs=model_runs,
                       save_path=save_path,
                       early_stop=True,
                       early_step=2)


def test_initial_point_generator(save_path):
    initial_point_generators = ['lhs', 'sobol', 'halton', 'hammersly', 'grid', 'random']
    # %% Load dataset
    dataset = Dataset()
    dataset.load("octis/preprocessed_datasets/M10")

    # %% Load model
    model = LDA()

    # %% Set model hyperparameters (not optimized by BO)
    model.hyperparameters.update({"num_topics": 25, "iterations": 200})
    model.partitioning(False)

    # %% Choose of the metric function to optimize
    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi = Coherence(metric_parameters)

    # %% Create search space for optimization
    search_space = {
        "alpha": Real(low=0.001, high=5.0),
        "eta": Real(low=0.001, high=5.0)
    }
    # %% Optimize the function npmi using Bayesian Optimization
    for initial_point_generator in initial_point_generators:
        save_path_specific = save_path + '/' + initial_point_generator
        optimizer = Optimizer()
        optimizer.optimize(model, dataset, npmi, search_space,
                           number_of_call=10,
                           model_runs=model_runs,
                           save_path=save_path_specific,
                           n_random_starts=5,
                           initial_point_generator=initial_point_generator)


# %%
def test_extra_metrics(save_path):
    # %% Load dataset
    dataset = Dataset()
    dataset.load("octis/preprocessed_datasets/M10")

    # %% Load model
    model = LDA()

    # %% Set model hyperparameters (not optimized by BO)
    model.hyperparameters.update({"num_topics": 25, "iterations": 200})
    model.partitioning(False)

    # %% Choose of the metric function to optimize
    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi = Coherence(metric_parameters)

    # %% Choose the extra metric to compute
    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi2 = Coherence(metric_parameters)
    # %% Create search space for optimization
    search_space = {
        "alpha": Real(low=0.001, high=5.0),
        "eta": Real(low=0.001, high=5.0)
    }
    # %% Optimize the function npmi using Bayesian Optimization (simple Optimization)
    optimizer = Optimizer()
    optimizer.optimize(model, dataset, npmi, search_space,
                       number_of_call=number_of_call,
                       model_runs=model_runs,
                       save_path=save_path,
                       extra_metrics=[npmi2],
                       save_models=False,
                       plot_best_seen=True,
                       plot_model=True)


def test_extra_metrics_resume(save_path):
    # %% Load dataset
    dataset = Dataset()
    dataset.load("octis/preprocessed_datasets/M10")

    # %% Load model
    model = LDA()

    # %% Set model hyperparameters (not optimized by BO)
    model.hyperparameters.update({"num_topics": 25, "iterations": 200})
    model.partitioning(False)

    # %% Choose of the metric function to optimize
    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi = Coherence(metric_parameters)

    # %% Choose the extra metric to compute
    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi2 = Coherence(metric_parameters)
    # %% Create search space for optimization
    search_space = {
        "alpha": Real(low=0.001, high=5.0),
        "eta": Real(low=0.001, high=5.0)
    }
    # %% Optimize the function npmi using Bayesian Optimization (simple Optimization)
    optimizer = Optimizer()
    BestObject = optimizer.optimize(model, dataset, npmi, search_space,
                                    number_of_call=number_of_call,
                                    model_runs=model_runs,
                                    save_path=save_path,
                                    extra_metrics=[npmi2])
    # %% Resume the optimization
    path = BestObject.name_json
    optimizer = Optimizer()
    optimizer.resume_optimization(path, extra_evaluations=3)


def test_extra_metrics_initial_input_points(save_path):
    # %% Load dataset
    dataset = Dataset()
    dataset.load("octis/preprocessed_datasets/M10")

    # %% Load model
    model = LDA()

    # %% Set model hyperparameters (not optimized by BO)
    model.hyperparameters.update({"num_topics": 25, "iterations": 200})
    model.partitioning(False)

    # %% Choose of the metric function to optimize
    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi = Coherence(metric_parameters)

    # %% Create search space for optimization
    search_space = {
        "alpha": Real(low=0.001, high=5.0),
        "eta": Real(low=0.001, high=5.0)
    }

    # %% inizialization of x0 as a dict
    x0 = {"eta": [0.1, 0.5, 0.5], "alpha": [0.5, 0.1, 0.5]}

    # %% Optimize the function npmi using Bayesian Optimization (simple Optimization)
    optimizer = Optimizer()
    optimizer.optimize(model, dataset, npmi, search_space,
                       number_of_call=number_of_call,
                       x0=x0,
                       y0=[-0.1, 0.1, -0.1],
                       plot_best_seen=True,
                       plot_model=True,
                       model_runs=model_runs,
                       save_path=save_path + 'test1/')
    # %% Optimize the function npmi using Bayesian Optimization (simple Optimization)
    optimizer = Optimizer()
    optimizer.optimize(model, dataset, npmi, search_space,
                       number_of_call=number_of_call,
                       x0=x0,
                       plot_best_seen=True,
                       plot_model=True,
                       model_runs=model_runs,
                       save_path=save_path + 'test2/')


# %% main function to test all the functions
def main():
    save_path = 'opt_tests/test_simple_optimization/'
    test_simple_optimization(save_path)

    save_path = 'opt_tests/test_resume_optimization/'
    test_resume_optimization(save_path)

    save_path = 'opt_tests/test_optimization_graphics/'
    test_optimization_graphics(save_path)

    save_path = 'opt_tests/test_optimization_acq_function/'
    test_optimization_acq_function(save_path)

    save_path = 'opt_tests/test_optimization_surrogate_model/'
    test_optimization_surrogate_model(save_path)

    save_path = 'opt_tests/test_early_stop/'
    test_early_stop(save_path)

    save_path = 'opt_tests/test_initial_point_generator/'
    test_initial_point_generator(save_path)

    save_path = 'opt_tests/test_extra_metrics/'
    test_extra_metrics(save_path)

    save_path = 'opt_tests/test_extra_metrics_resume/'
    test_extra_metrics_resume(save_path)

    save_path = 'opt_tests/test_extra_metrics_initial_input_points/'
    test_extra_metrics_initial_input_points(save_path)


if __name__ == "__main__":
    main()
