# Utils
from skopt.space.space import Real, Integer
from skopt.utils import dimensions_aslist
from optimization.optimization_result import Best_evaluation
from optimization.optimizer_tool import plot_bayesian_optimization
from optimization.optimizer_tool import plot_boxplot
from optimization.optimizer_tool import median_number
from optimization.csv_creator import save_csv
from models.model import save_model_output

import time
import numpy as np
from skopt import dump, load
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
import os
from pathlib import Path  # Path(path).mkdir(parents=True, exist_ok=True)

# Models
from optimization.gp_minimizer import gp_minimizer as gp_minimizer_function
from optimization.forest_minimizer import forest_minimizer as forest_minimizer_function
from optimization.random_minimizer import random_minimizer as random_minimizer_function
from skopt import gp_minimize, forest_minimize, dummy_minimize

# Kernel
from sklearn.gaussian_process.kernels import Matern

Matern_kernel_3 = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)

# Initialize default parameters
default_parameters = {
    'n_calls': 100,
    'optimization_runs': 3,
    'model_runs': 10,
    'n_random_starts': 10,  # Should be one for dimension (at least)
    'minimizer': gp_minimize,
    'acq_func': "LCB",
    'kernel': Matern_kernel_3,
    'random_state': None,
    'noise': None,
    'verbose': False,
    'n_points': 10000,
    'base_estimator': 'RF',
    'kappa': 1.96,
    'alpha': 1e-10,
    'x0': [None],
    'y0': [None],
    'time_x0': None,
    'xi': 1.96,
    'n_jobs': 1,
    'model_queue_size': None,
    'optimization_type': 'Maximize',
    'extra_metrics': [],
    'save_models': False,
    'save': False,
    'save_step': 1,
    'save_name': "result",
    'save_path': None,  # where to save all file (log, txt, plot, etc)
    'early_stop': False,
    'early_step': 10,
    'plot_optimization': False,
    'plot_model': False,
    'plot_name': "Bayesian optimization plot",
    'log_scale_plot': False
}


class Optimizer():
    """
    Optimizer optimize hyperparameters to build topic models
    """

    # Values of hyperparameters and metrics for each iteration
    _iterations = []

    topk = 10  # if False the topk words will not be computed
    topic_word_matrix = True  # if False the matrix will not be computed
    topic_document_matrix = True  # if False the matrix will not be computed

    def __init__(self, model, dataset, metric, search_space, optimization_parameters={}):
        """
        Inititalize the optimizer for the model

        Parameters
        ----------
        model : model with hyperparameters to optimize
        metric : initialized metric to use for optimization
        search_space : a dictionary of hyperparameters to optimize
                       (each parameter is defined as a skopt space)
                       with the name of the hyperparameter given as key
        optimization_parameters : parameters of the search
        """
        self.model = model
        self.dataset = dataset
        self.metric = metric
        self.search_space = search_space
        self.current_call = 0
        self.current_optimization_run = 0

        if (optimization_parameters["save_path"][-1] != '/'):
            optimization_parameters["save_path"] = optimization_parameters["save_path"] + '/'

        self.optimization_parameters = optimization_parameters

        # Customize parameters update
        default_parameters.update(self.optimization_parameters)
        # Save parameters labels to use
        self.hyperparameters = list(sorted(self.search_space.keys()))

        self.extra_metrics = default_parameters["extra_metrics"]

        self.optimization_type = default_parameters['optimization_type']

        if ((default_parameters["save_models"] == True) and (default_parameters["save_path"] is not None)):
            model_path = default_parameters["save_path"] + "models/"
            Path(model_path).mkdir(parents=True, exist_ok=True)

        # Store the different value of the metric for each model_runs

        if (default_parameters["minimizer"] != forest_minimize):
            self.matrix_model_runs = np.zeros((default_parameters["n_calls"],
                                               default_parameters["optimization_runs"],
                                               default_parameters["model_runs"]))
        else:
            self.matrix_model_runs = np.zeros((default_parameters["n_calls"] + default_parameters["n_random_starts"],
                                               default_parameters["optimization_runs"],
                                               default_parameters["model_runs"]))

    def _objective_function(self, hyperparameters, path=None):
        """
        objective function to optimize

        Parameters
        ----------
        hyperparameters : dictionary of hyperparameters
                          (It's a list for real)
                          key: name of the parameter
                          value: skopt search space dimension

        Returns
        -------
        result : score of the metric to maximize
        """

        # Retrieve parameters labels
        params = {}
        for i in range(len(self.hyperparameters)):
            params[self.hyperparameters[i]] = hyperparameters[i]

        # Get the median of the metric score
        different_model_runs = []
        for i in range(default_parameters["model_runs"]):
            # Prepare model
            model_output = self.model.train_model(self.dataset, params, self.topk,
                                                  self.topic_word_matrix, self.topic_document_matrix)

            model_res = self.metric.score(model_output)
            self.matrix_model_runs[self.current_call, self.current_optimization_run, i] = model_res
            different_model_runs.append(model_res)

            # Save the models
            if (default_parameters["save_models"]):
                nome_giusto = str(self.current_call) + "_" + str(self.current_optimization_run) + "_" + str(
                    i)  # "<n_calls>_<optimization_runs>_<model_runs>"
                if path is None:
                    save_model_path = default_parameters["save_path"] + "models/" + nome_giusto
                if path is not None:
                    if path[-1] != '/':
                        path = path + '/'
                    save_model_path = path + nome_giusto
                    Path(path).mkdir(parents=True, exist_ok=True)

                save_model_output(model_output, save_model_path)

        result = median_number(different_model_runs)

        # Indici save
        if self.current_optimization_run + 1 == default_parameters["optimization_runs"]:
            # print( default_parameters["optimization_runs"] , "if" )
            self.current_call = self.current_call + 1
            self.current_optimization_run = 0
        else:
            # print( default_parameters["optimization_runs"] , "else" )
            self.current_optimization_run = self.current_optimization_run + 1

        # print(self.current_call,"_",self.current_optimization_run,"->",self.metric.score(model_output) )

        # Update metrics values for extra metrics
        metrics_values = {self.metric.__class__.__name__: result}
        iteration = [hyperparameters, metrics_values]
        for extra_metric in self.extra_metrics:

            extra_metric_name = extra_metric.__class__.__name__
            if extra_metric_name not in metrics_values:
                name = extra_metric_name
            else:
                i = 2
                name = extra_metric_name + " 2"
                while name in metrics_values:
                    i += 1
                    name = extra_metric_name + " " + str(i)

            metrics_values[name] = extra_metric.score(model_output)

        # Save iteration data
        self._iterations.append(iteration)

        if self.optimization_type == 'Maximize':
            result = - result

        #print("Model_runs ->", different_model_runs)
        #print("Mediana->", result)
        #print("Matrix->", self.matrix_model_runs)

        # Need to work only when optimization_runs is 1
        if default_parameters["plot_model"] and (default_parameters["optimization_runs"] == 1):
            default_parameters["plot_model"] = True

            if default_parameters["plot_name"].endswith(".png"):
                name = default_parameters["plot_name"]
            else:
                name = default_parameters["plot_name"] + ".png"

            if not default_parameters["plot_optimization"]:
                name_model_plot = name
            else:
                name_model_plot = name[:-4] + "_model.png"

            #print("name_model_plot->", name_model_plot)

            plot_boxplot(self.matrix_model_runs, name_model_plot, path=default_parameters["save_path"])
        

        return result

    def Bayesian_optimization(self, f,  # = self.self._objective_function,#
                              bounds,  # = params_space_list,#
                              minimizer=default_parameters["minimizer"],
                              number_of_call=default_parameters["n_calls"],
                              optimization_runs=default_parameters["optimization_runs"],
                              model_runs=default_parameters["model_runs"],
                              kernel=default_parameters["kernel"],
                              acq_func=default_parameters["acq_func"],
                              base_estimator_forest=default_parameters["base_estimator"],
                              random_state=default_parameters["random_state"],
                              noise_level=default_parameters["noise"],
                              alpha=default_parameters["alpha"],
                              kappa=default_parameters["kappa"],
                              X0=default_parameters["x0"],
                              Y0=default_parameters["y0"],
                              time_x0=default_parameters["time_x0"],
                              n_random_starts=default_parameters["n_random_starts"],
                              save=default_parameters["save"],
                              save_step=default_parameters["save_step"],
                              save_name=default_parameters["save_name"],
                              save_path=default_parameters["save_path"],
                              early_stop=default_parameters["early_stop"],
                              early_step=default_parameters["early_step"],
                              plot_optimization=default_parameters["plot_optimization"],
                              plot_model=default_parameters["plot_model"],
                              plot_name=default_parameters["plot_name"],
                              log_scale_plot=default_parameters["log_scale_plot"],
                              verbose=default_parameters["verbose"],
                              n_points=default_parameters["n_points"],
                              xi=default_parameters["xi"],
                              n_jobs=default_parameters["n_jobs"],
                              model_queue_size=default_parameters["model_queue_size"]):
        """
            Bayesian_optimization

            Parameters
            ----------
            f : Function to minimize. 
                Should take a single list of parameters and return the objective value.

            bounds : List of search space dimensions. Each search dimension can be defined either as
                    - (lower_bound, upper_bound) tuple (for Real or Integer dimensions),
                    - (lower_bound, upper_bound, "prior") tuple (for Real dimensions),
                    - list of categories (for Categorical dimensions), or
                    - instance of a Dimension object (Real, Integer or Categorical).
                    Note: The upper and lower bounds are inclusive for Integer.

            minimizer : The base estimator to use for optimization.
                        -gp_minimize
                        -dummy_minimize
                        -forest_minimize

            number_of_call : Number of calls to f

            optimization_runs : Number of different run of a single Bayesian Optimization
                                [min = 3]

            model_runs: Number of different evaluation of the function in the same point
                        and with the same hyperparameters. Usefull with a lot of noise.
            
            kernel : The kernel specifying the covariance function of the GP.
            
            acq_func : Function to minimize over the minimizer prior. Can be either
                        - "LCB" for lower confidence bound.
                        - "EI" for negative expected improvement.
                        - "PI" for negative probability of improvement.
                        - "gp_hedge" probabilistically choose one of the above three acquisition 
                            functions at every iteration[only if minimizer == gp_minimize]
                        - "EIps" for negated expected improvement.
                        - "PIps" for negated probability of improvement.
            
            base_estimator_forest : The regressor to use as surrogate model. Can be either
                                    - "RF" for random forest regressor
                                    - "ET" for extra trees regressor
                                    instance of regressor with support for return_std in its predict method.

            random_state : Set random state to something other than None for reproducible results.
            
            noise_level : If set to “gaussian”, then it is assumed that y is a noisy estimate 
                        of f(x) where the noise is gaussian.
            
            alpha : Value added to the diagonal of the kernel matrix during fitting. 
                    Larger values correspond to increased noise level in the observations and 
                    reduce potential numerical issue during fitting. If an array is passed, 
                    it must have the same number of entries as the data used for fitting and is 
                    used as datapoint-dependent noise level. Note that this is equivalent to adding
                    a WhiteKernel with c=alpha. Allowing to specify the noise level directly as a 
                    parameter is mainly for convenience and for consistency with Ridge.
            
            kappa : Controls how much of the variance in the predicted values should be taken into account. 
                    If set to be very high, then we are favouring exploration over exploitation and vice versa. 
                    Used when the acquisition is "LCB"
            
            X0 : Initial input points.
            
            Y0 : Evaluation of initial input points.

            time_x0 : Time to evaluate x0 and y0
            
            n_random_starts : Number of evaluations of f with random points before 
                            approximating it with minimizer
            
            save : [boolean] Save the Bayesian Optimization in a .pkl and .cvs file 
            
            save_step : Integer interval after which save the .pkl file
            
            save_name : Name of the .pkl and .cvs file saved.
                        Useless if save is False.
            
            save_path : Path where .pkl, plot and result will be saved.
            
            early_stop : [boolean] Early stop policy.
                        It will stop an interaction if it doesn't
                        improve for early_step evaluations.
            
            early_step : Integer interval after which a current optimization run
                        is stopped if it doesn't improve.
            
            plot_optimization : [boolean] Plot the convergence of the Bayesian optimization 
                    process, showing mean and standard deviation of the different
                    optimization runs. 
                    If save is True the plot is update every save_step evaluations.

            plot_model: [boolean] Plot the mean and standard deviation of the different
                    model runs. 
                    If save is True the plot is update every save_step evaluations.
            
            plot_name : Name of the .png file where the plot is saved.
            
            log_scale_plot : [boolean] If True the "y_axis" of the plot
                            is set to log_scale
            
            verbose : Control the verbosity. It is advised to set the verbosity to True for long optimization runs.
            
            n_points : Number of points to sample to determine the next “best” point. 
                    Useless if acq_optimizer is set to "lbfgs".
            
            xi  : Controls how much improvement one wants over the previous best values. 
                Used when the acquisition is either "EI" or "PI".
            
            n_jobs : Number of cores to run in parallel while running the lbfgs optimizations 
                    over the acquisition function. Valid only when acq_optimizer is set to “lbfgs.”
                    Defaults to 1 core. If n_jobs=-1, then number of jobs is set to number of cores.
            
            model_queue_size : Keeps list of models only as long as the argument given.
                            In the case of None, the list has no capped length.
            
            Returns
            -------
            res : List of different Bayesian Optimization run.
                Important attributes of each element are:
                - x [list]: location of the minimum.
                - fun [float]: function value at the minimum.
                - models: surrogate models used for each optimization run.
                - x_iters [list of lists]: location of function evaluation for each optimization run.
                - func_vals [array]: function value for each optimization run.
                - space [Space]: the optimization space.
                - specs [dict]`: the call specifications.
                - rng [RandomState instance]: State of the random state at the end of minimization.
        """

        if number_of_call <= 0:
            print("Error: number_of_call can't be <= 0")
            return None

        # dimensioni = len( bounds )
        checkpoint_saver = [None] * optimization_runs

        if X0 == [None]:
            x0 = [None] * optimization_runs
        else:
            x0 = X0

        if Y0 == [None]:
            y0 = [None] * optimization_runs
        else:
            y0 = Y0

        if default_parameters["minimizer"] == gp_minimize:
            minimizer_stringa = "gp_minimize"
        elif default_parameters["minimizer"] == dummy_minimize:
            minimizer_stringa = "random_minimize"
        elif default_parameters["minimizer"] == forest_minimize:
            minimizer_stringa = "forest_minimize"
        else:
            minimizer_stringa = "None"

        if save and save_path is not None:
            Path(save_path).mkdir(parents=True, exist_ok=True)

        print("------------------------------------------")
        print("------------------------------------------")
        print("Bayesian optimization parameters:\n-n_calls: ", default_parameters["n_calls"],
              "\n-optimization_runs: ", default_parameters["optimization_runs"],
              "\n-model_runs: ", default_parameters["model_runs"],
              "\n-n_random_starts: ", default_parameters["n_random_starts"],
              "\n-minimizer: ", minimizer_stringa)
        if default_parameters["minimizer"] != dummy_minimize:
            print("-acq_func: ", default_parameters["acq_func"])
        if default_parameters["minimizer"] == gp_minimize:
            print("-kernel: ", default_parameters["kernel"])
        print("------------------------------------------")

        # Dummy Minimize
        if minimizer == dummy_minimize:
            return random_minimizer_function(f=f, bounds=bounds,
                                    number_of_call=number_of_call,
                                    optimization_runs=optimization_runs,
                                    model_runs=model_runs,
                                    random_state=random_state,
                                    x0=x0,
                                    y0=y0,
                                    time_x0=time_x0,
                                    n_random_starts=n_random_starts,
                                    save=save,
                                    save_step=save_step,
                                    save_name=save_name,
                                    save_path=save_path,
                                    early_stop=early_stop,
                                    early_step=early_step,
                                    plot_optimization=plot_optimization,
                                    plot_model=plot_model,
                                    plot_name=plot_name,
                                    log_scale_plot=log_scale_plot,
                                    verbose=verbose,
                                    model_queue_size=model_queue_size,
                                    checkpoint_saver=checkpoint_saver,
                                    dataset_name=self.dataset.get_metadata()["info"]["name"],
                                    hyperparameters_name=self.hyperparameters,
                                    metric_name=self.metric.__class__.__name__,
                                    num_topic=self.model.hyperparameters['num_topics'],
                                    maximize=(self.optimization_type == 'Maximize'))

        # Forest Minimize
        elif minimizer == forest_minimize:
            return forest_minimizer_function(f=f, bounds=bounds,
                                    number_of_call=number_of_call,
                                    optimization_runs=optimization_runs,
                                    model_runs=model_runs,
                                    acq_func=acq_func,
                                    base_estimator_forest=base_estimator_forest,
                                    random_state=random_state,
                                    kappa=kappa,
                                    x0=x0,
                                    y0=y0,
                                    time_x0=time_x0,
                                    n_random_starts=n_random_starts,
                                    save=save,
                                    save_step=save_step,
                                    save_name=save_name,
                                    save_path=save_path,
                                    early_stop=early_stop,
                                    early_step=early_step,
                                    plot_optimization=plot_optimization,
                                    plot_model=plot_model,
                                    plot_name=plot_name,
                                    log_scale_plot=log_scale_plot,
                                    verbose=verbose,
                                    n_points=n_points,
                                    xi=xi,
                                    n_jobs=n_jobs,
                                    model_queue_size=model_queue_size,
                                    checkpoint_saver=checkpoint_saver,
                                    dataset_name=self.dataset.get_metadata()["info"]["name"],
                                    hyperparameters_name=self.hyperparameters,
                                    metric_name=self.metric.__class__.__name__,
                                    num_topic=self.model.hyperparameters['num_topics'],
                                    maximize=(self.optimization_type == 'Maximize'))

        # GP Minimize
        elif minimizer == gp_minimize:
            return gp_minimizer_function(f=f, bounds=bounds,
                                number_of_call=number_of_call,
                                optimization_runs=optimization_runs,
                                model_runs=model_runs,
                                kernel=kernel,
                                acq_func=acq_func,
                                random_state=random_state,
                                noise_level=noise_level,  # attenzione
                                alpha=alpha,
                                x0=x0,
                                y0=y0,
                                time_x0=time_x0,
                                n_random_starts=n_random_starts,
                                save=save,
                                save_step=save_step,
                                save_name=save_name,
                                save_path=save_path,
                                early_stop=early_stop,
                                early_step=early_step,
                                plot_optimization=plot_optimization,
                                plot_model=plot_model,
                                plot_name=plot_name,
                                log_scale_plot=log_scale_plot,
                                verbose=verbose,
                                model_queue_size=model_queue_size,
                                checkpoint_saver=checkpoint_saver,
                                dataset_name=self.dataset.get_metadata()["info"]["name"],
                                hyperparameters_name=self.hyperparameters,
                                metric_name=self.metric.__class__.__name__,
                                num_topic=self.model.hyperparameters['num_topics'],
                                maximize=(self.optimization_type == 'Maximize'))

        else:
            print("Error. Not such minimizer: ", minimizer)

    def optimize(self):
        """
        Optimize the hyperparameters of the model

        Parameters
        ----------


        Returns
        -------
        result : Best_evaluation object
        """
        self._iterations = []

        # Save parameters labels to use
        self.hyperparameters = list(sorted(self.search_space.keys()))
        params_space_list = dimensions_aslist(self.search_space)

        # Customize parameters update
        default_parameters.update(self.optimization_parameters)
        # print("default parameters ", default_parameters )
        self.extra_metrics = default_parameters["extra_metrics"]

        self.optimization_type = default_parameters['optimization_type']

        # Optimization call
        optimize_result = self.Bayesian_optimization(
            f=self._objective_function,
            bounds=params_space_list,
            minimizer=default_parameters["minimizer"],
            number_of_call=default_parameters["n_calls"],
            optimization_runs=default_parameters["optimization_runs"],
            kernel=default_parameters["kernel"],
            acq_func=default_parameters["acq_func"],
            base_estimator_forest=default_parameters["base_estimator"],
            random_state=default_parameters["random_state"],
            noise_level=default_parameters["noise"],
            alpha=default_parameters["alpha"],
            kappa=default_parameters["kappa"],
            X0=default_parameters["x0"],
            Y0=default_parameters["y0"],
            time_x0=default_parameters["time_x0"],
            n_random_starts=default_parameters["n_random_starts"],
            save=default_parameters["save"],
            save_step=default_parameters["save_step"],
            save_name=default_parameters["save_name"],
            save_path=default_parameters["save_path"],
            early_stop=default_parameters["early_stop"],
            early_step=default_parameters["early_step"],
            plot_optimization=default_parameters["plot_optimization"],
            plot_model=default_parameters["plot_model"],
            plot_name=default_parameters["plot_name"],
            log_scale_plot=default_parameters["log_scale_plot"],
            verbose=default_parameters["verbose"],
            n_points=default_parameters["n_points"],
            xi=default_parameters["xi"],
            n_jobs=default_parameters["n_jobs"],
            model_queue_size=default_parameters["model_queue_size"])

        # To have the right result
        if self.optimization_type == 'Maximize':
            for i in range(len(optimize_result)):
                optimize_result[i].fun = - optimize_result[i].fun
                for j in range(len(optimize_result[i].func_vals)):
                    optimize_result[i].func_vals[j] = - optimize_result[i].func_vals[j]

            if default_parameters["plot_optimization"]:
                plot_bayesian_optimization(list_of_res=optimize_result,
                                           name_plot=default_parameters["plot_name"],
                                           log_scale=default_parameters["log_scale_plot"],
                                           path=default_parameters["save_path"],
                                           conv_min=False)

        # Create Best_evaluation object from optimization results
        result = Best_evaluation(self.hyperparameters,
                                 optimize_result,
                                 self.optimization_type == 'Maximize',
                                 # Maximize = (self.optimization_type == 'Maximize')
                                 self._iterations,
                                 self.metric.__class__.__name__)

        return result
