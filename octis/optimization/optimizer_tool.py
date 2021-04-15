import matplotlib.pyplot as plt
import numpy as np
from skopt.learning import GaussianProcessRegressor, RandomForestRegressor, ExtraTreesRegressor
from skopt import Optimizer as skopt_optimizer
from skopt.utils import dimensions_aslist
import os
import importlib
import sys
import octis.configuration.defaults as defaults
from pathlib import Path

framework_path = Path(os.path.dirname(os.path.realpath(__file__)))
framework_path = str(framework_path.parent)


def importClass(class_name, module_name, module_path):
    """
    Import a class runtime based on its module and name

    :param class_name: name of the class
    :type class_name: str
    :param module_name: name of the module
    :type module_name: str
    :param module_path: absolute path to the module
    :type module_path: str
    :return: class object
    :rtype: class
    """
    spec = importlib.util.spec_from_file_location(
        module_name, module_path, submodule_search_locations=[])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    importlib.invalidate_caches()
    imported_class = getattr(module, class_name)

    return imported_class


def load_model(optimization_object):
    """
    Load the topic model for the resume of the optimization

    :param optimization_object: dictionary of optimization attributes saved in the jaon file
    :type optimization_object: dict
    :return: topic model used during the BO.
    :rtype: object model
    """

    model_parameters = optimization_object['model_attributes']
    use_partitioning = optimization_object['use_partitioning']

    model_name = optimization_object['model_name']
    module_path = os.path.join(framework_path, "models")
    module_path = os.path.join(module_path, model_name + ".py")
    model = importClass(model_name, model_name, module_path)
    model_instance = model()
    model_instance.hyperparameters.update(model_parameters)
    model_instance.use_partitions = use_partitioning

    return model_instance


def select_metric(metric_parameters, metric_name):
    """
    Select the metric for the resume of the optimization

    :param metric_parameters: metric parameters
    :type metric_parameters: list
    :param metric_name: name of the metric
    :type metric_name: str
    :return: metric
    :rtype: metric object
    """
    module_path = os.path.join(framework_path, "evaluation_metrics")
    module_name = defaults.metric_parameters[metric_name]["module"]
    module_path = os.path.join(module_path, module_name + ".py")
    Metric = importClass(metric_name, metric_name, module_path)
    metric = Metric(**metric_parameters)

    return metric


def choose_optimizer(optimizer):
    """
    Choose a surrogate model for Bayesian Optimization

    :param optimizer: list of setting of the BO experiment
    :type optimizer: Optimizer
    :return: surrogate model
    :rtype: scikit object
    """
    params_space_list = dimensions_aslist(optimizer.search_space)
    estimator = None
    # Choice of the surrogate model
    # Random forest
    if optimizer.surrogate_model == "RF":
        estimator = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=3, random_state=optimizer.random_state)
    # Extra Tree
    elif optimizer.surrogate_model == "ET":
        estimator = ExtraTreesRegressor(
            n_estimators=100, min_samples_leaf=3, random_state=optimizer.random_state)
    # GP Minimize
    elif optimizer.surrogate_model == "GP":
        estimator = GaussianProcessRegressor(
            kernel=optimizer.kernel, random_state=optimizer.random_state)
        # Random Search
    elif optimizer.surrogate_model == "RS":
        estimator = "dummy"

    if estimator == "dummy":
        opt = skopt_optimizer(params_space_list, base_estimator=estimator,
                              acq_func=optimizer.acq_func,
                              acq_optimizer='sampling',
                              initial_point_generator=optimizer.initial_point_generator,
                              random_state=optimizer.random_state)
    else:
        opt = skopt_optimizer(params_space_list, base_estimator=estimator,
                              acq_func=optimizer.acq_func,
                              acq_optimizer='sampling',
                              n_initial_points=optimizer.n_random_starts,
                              initial_point_generator=optimizer.initial_point_generator,
                              # work only for version skopt 8.0!!!
                              acq_optimizer_kwargs={
                                  "n_points": 10000, "n_restarts_optimizer": 5, "n_jobs": 1},
                              acq_func_kwargs={"xi": 0.01, "kappa": 1.96},
                              random_state=optimizer.random_state)
    return opt


def convergence_res(values, optimization_type="minimize"):
    """
    Compute the list of values to plot the convergence plot (i.e. the best seen at each iteration)

    :param values: the result(s) for which to compute the convergence trace.
    :type values: list
    :param optimization_type: "minimize" if the problem is a minimization problem, "maximize" otherwise
    :type optimization_type: str
    :return: a list with the best min seen for each iteration
    :rtype: list
    """

    values2 = values.copy()

    if optimization_type == "minimize":
        for i in range(1, len(values2)):
            if values2[i] > values2[i - 1]:
                values2[i] = values2[i - 1]
    else:
        for i in range(1, len(values2)):
            if values2[i] < values2[i - 1]:
                values2[i] = values2[i - 1]
    return values2


def early_condition(values, n_stop, n_random):
    """
    Compute the early-stop criterium to stop or not the optimization.

    :param values: values obtained by Bayesian Optimization
    :type values: list
    :param n_stop: Range of points without improvement
    :type n_stop: int
    :param n_random: Random starting points
    :type n_random: int
    :return: 'True' if early stop condition reached, 'False' otherwise
    :rtype: bool
    """
    n_min_len = n_stop + n_random

    if len(values) >= n_min_len:
        values = convergence_res(values, optimization_type="minimize")
        worst = values[len(values) - n_stop]
        best = values[-1]
        diff = worst - best
        if diff == 0:
            return True

    return False


def plot_model_runs(model_runs, current_call, name_plot):
    """
    Save a boxplot of the data (Works only when optimization_runs is 1).

    :param model_runs: dictionary of all the model runs.
    :type model_runs: dict
    :param current_call: number of calls computed by BO
    :type current_call: int
    :param name_plot: Name of the plot
    :type name_plot: str
    """
    values = [model_runs["iteration_" + str(i)] for i in range(current_call + 1)]

    plt.ioff()
    plt.xlabel('number of calls')
    plt.grid(True)
    plt.boxplot(values)

    plt.savefig(name_plot + ".png")

    plt.close()


def plot_bayesian_optimization(values, name_plot,
                               log_scale=False, conv_max=True):
    """
    Save a convergence plot of the result of a Bayesian_optimization.

    :param values: List of objective function values
    :type values: list
    :param name_plot: Name of the plot
    :type name_plot: str
    :param log_scale: 'True' if you want a log scale for y-axis, 'False' otherwise
    :type log_scale: bool, optional
    :param conv_max: 'True' for a minimization problem, 'False' for a maximization problem
    :type conv_max: bool, optional
    """
    if conv_max:
        # minimization problem -->maximization problem
        values = [-val for val in values]
        media = convergence_res(values, optimization_type="maximize")
        xlabel = 'max f(x) after n calls'
    else:
        # minimization problem
        media = convergence_res(values, optimization_type="minimize")
        xlabel = 'min f(x) after n calls'

    array = [i for i in range(len(media))]

    plt.ioff()
    plt.plot(array, media, color='blue', label="res")

    if log_scale:
        plt.yscale('log')

    plt.ylabel(xlabel)
    plt.xlabel('Number of calls n')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True)

    plt.savefig(name_plot + ".png")

    plt.close()


def convert_type(obj):
    """
    Convert a numpy object to a python object

    :param obj: object to be checked
    :type obj: numpy object
    :return: python object
    :rtype: python object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def check_instance(obj):
    """
    Check if a specific object con be inserted in the json file.

    :param obj: an object of the optimization to be saved
    :type obj: [str,float, int, bool, etc.]
    :return: 'True' if the object can be inserted in a json file, 'False' otherwise
    :rtype: bool
    """
    types = [str, float, int, bool]

    for t in types:
        if isinstance(obj, t):
            return True

    return False


def save_search_space(search_space):
    """
    Save the search space in the json file

    :param search_space: dictionary of the search space (scikit-optimize object)
    :type search_space: dict
    :return: dictionary for the seach space, which can be saved in a json file
    :rtype: dict
    """
    from skopt.space.space import Real, Categorical, Integer

    ss = dict()
    for key in list(search_space.keys()):
        if type(search_space[key]) == Real:
            ss[key] = ['Real', search_space[key].bounds, search_space[key].prior]
        elif type(search_space[key]) == Integer:
            ss[key] = ['Integer', search_space[key].bounds, search_space[key].prior]
        elif type(search_space[key]) == Categorical:
            ss[key] = ['Categorical', search_space[key].categories, search_space[key].prior]

    return ss


def load_search_space(search_space):
    """
    Load the search space from the json file

    :param search_space: dictionary of the search space (insertable in a json file)
    :type dict:
    :return: dictionary for the search space (for scikit optimize)
    :rtype: dict
    """
    from skopt.space.space import Real, Categorical, Integer

    ss = dict()
    for key in list(search_space.keys()):
        if search_space[key][0] == 'Real':
            ss[key] = Real(low=search_space[key][1][0], high=search_space[key][1][1], prior=search_space[key][2])
        elif search_space[key][0] == 'Integer':
            ss[key] = Integer(low=search_space[key][1][0], high=search_space[key][1][1], prior=search_space[key][2])
        elif search_space[key][0] == 'Categorical':
            ss[key] = Categorical(categories=search_space[key][1])

    return ss

