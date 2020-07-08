from skopt.space.space import Real, Integer
from skopt.utils import dimensions_aslist
from optimization.optimization_result import Best_evaluation
from optimization.stopper import MyCustomEarlyStopper
import optimization.optimizer_tool as tool
from optimization.csv_creator import save_csv as save_csv
from models.model import save_model_output as save_model_output

import time
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from skopt import dump, load
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
from pathlib import Path #Path(path).mkdir(parents=True, exist_ok=True)


#Acquisition function
from skopt.acquisition import gaussian_ei
from skopt.acquisition import gaussian_lcb
from skopt.acquisition import gaussian_pi

#Kernel
from skopt.plots import plot_convergence
from skopt.callbacks import EarlyStopper
from skopt import Optimizer as skopt_optimizer
from skopt.learning import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, DotProduct,
                                              ConstantKernel, ExpSineSquared)

kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
           ConstantKernel(0.1, (0.01, 10.0))
               * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
           1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                        nu=0.5),
            1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                        nu=1.5),
            1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                        nu=2.5),
            1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                length_scale_bounds=(0.1, 10.0),
                                periodicity_bounds=(1.0, 10.0))]


#Models
from functools import partial
from skopt.benchmarks import branin as _branin
from skopt import gp_minimize, forest_minimize, dummy_minimize


# Initialize default parameters
default_parameters = {
    'n_calls': 100,
    'different_iteration': 10, 
    'n_random_starts': 10, #Should be one for dimension (at least)
    'minimizer': gp_minimize, 
    'acq_func': "LCB",
    'kernel': kernels[3], 
    'random_state': None,
    'noise': None,
    'verbose': False,
    'n_points': 10000,
    'base_estimator': 'RF',
    'kappa': 1.96,
    'alpha': 1e-10,
    'x0': [None],
    'y0': [None],
    'time_x0' : None,
    'xi': 1.96,
    'n_jobs': 1,
    'model_queue_size': None,
    'optimization_type': 'Maximize',
    'extra_metrics': [],
    'save_models': False,
    'save': False, 
    'save_step': 1, 
    'save_name': "result",
    'save_path': None, #where to save all file (log, txt, plot, etc)
    'early_stop': False, 
    'early_step': 10, 
    'plot': False, 
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
        self.actual_call = 0
        self.actual_iteration = 0

        if( optimization_parameters["save_path"][-1] != '/' ):
            optimization_parameters["save_path"] = optimization_parameters["save_path"]+'/'

        self.optimization_parameters = optimization_parameters

        # Customize parameters update
        default_parameters.update(self.optimization_parameters)
        # Save parameters labels to use
        self.hyperparameters = list(sorted(self.search_space.keys()))

        self.extra_metrics = default_parameters["extra_metrics"]

        self.optimization_type = default_parameters['optimization_type']

        if( (default_parameters["save_models"] == True) and (default_parameters["save_path"] != None) ):
            model_path = default_parameters["save_path"] + "models/"
            Path(model_path).mkdir(parents=True, exist_ok=True)
        
    def _objective_function(self, hyperparameters, path = None):
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

        # Prepare model
        model_output = self.model.train_model(
            self.dataset,
            params,
            self.topk,
            self.topic_word_matrix,
            self.topic_document_matrix)

        #Save the models
        if( default_parameters["save_models"] ):
            nome_giusto = str(self.actual_call) + "_" + str(self.actual_iteration) # "/nIterazione_nRun"
            if( path == None ):
                save_model_path = default_parameters["save_path"] + "models/" + nome_giusto
            if( path != None ):
                if( path[-1] != '/' ):
                    path = path+'/'
                save_model_path = path + nome_giusto
                Path(path).mkdir(parents=True, exist_ok=True)

            save_model_output(model_output, save_model_path)
            if( self.actual_iteration +1 == default_parameters["different_iteration"] ):  #'n_calls': 100
                #print( default_parameters["different_iteration"] , "if" )
                self.actual_call = self.actual_call + 1
                self.actual_iteration = 0
            else:
                #print( default_parameters["different_iteration"] , "else" )
                self.actual_iteration = self.actual_iteration + 1


        # Get metric score
        result = self.metric.score(model_output)

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
                    name = extra_metric_name + " "+str(i)

            metrics_values[name] = extra_metric.score(model_output)

        # Save iteration data
        self._iterations.append(iteration)

        if self.optimization_type == 'Maximize':
            result = - result

        return result


    def Bayesian_optimization(self, f ,#= self.self._objective_function,#
                              bounds,#= params_space_list,#
                              minimizer=default_parameters["minimizer"],
                              number_of_call=default_parameters["n_calls"],
                              different_iteration = default_parameters["different_iteration"],
                            kernel=default_parameters["kernel"],
                            acq_func=default_parameters["acq_func"],
                            base_estimator_forest=default_parameters["base_estimator"],
                            random_state = default_parameters["random_state"],
                            noise_level = default_parameters["noise"],
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
                            plot = default_parameters["plot"],
                            plot_name = default_parameters["plot_name"],
                            log_scale_plot = default_parameters["log_scale_plot"],
                            verbose = default_parameters["verbose"],
                            n_points = default_parameters["n_points"],
                            xi  = default_parameters["xi"],
                            n_jobs = default_parameters["n_jobs"],
                            model_queue_size = default_parameters["model_queue_size"]
        ):
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

            different_iteration : Number of different iteration of a single Bayesian Optimization
                                [min = 3]
            
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
            
            early_step : Integer interval after which a current iteration
                        is stopped if it doesn't improve.
            
            plot : [boolean] Plot the convergence of the Bayesian optimization 
                    process, showing mean and standard deviation of the different
                    iterations. 
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
                - models: surrogate models used for each iteration.
                - x_iters [list of lists]: location of function evaluation for each iteration.
                - func_vals [array]: function value for each iteration.
                - space [Space]: the optimization space.
                - specs [dict]`: the call specifications.
                - rng [RandomState instance]: State of the random state at the end of minimization.
        
        """  
    
        if( number_of_call <= 0 ):
            print("Error: number_of_call can't be <= 0")
            return None

        if( different_iteration <= 2 ): 
            print("Error: different iteration should be 3 or more")
            return None
        
        res = []
        #dimensioni = len( bounds )
        checkpoint_saver = [None] * different_iteration

        if( X0 == [None] ):
            x0 = [None]*different_iteration
        else:
            x0 = X0
            
        if( Y0 == [None] ):
            y0 = [None]*different_iteration
        else:
            y0 = Y0

        if( default_parameters["minimizer"] == gp_minimize ):
            minimizer_stringa = "gp_minimize"
        
        if( default_parameters["minimizer"] == dummy_minimize ):
            minimizer_stringa = "random_minimize"

        if( default_parameters["minimizer"] == forest_minimize ):
            minimizer_stringa = "forest_minimize"

        if( (save == True) and (save_path != None) ):
            Path(save_path).mkdir(parents=True, exist_ok=True)


        print("------------------------------------------")
        print("------------------------------------------")
        print("Bayesian optimization parameters:\n-n_calls: ",default_parameters["n_calls"],
            "\n-different_iteration: ",default_parameters["different_iteration"],
            "\n-n_random_starts: ",default_parameters["n_random_starts"],
            "\n-minimizer: ",minimizer_stringa,
            "\n-acq_func: ",default_parameters["acq_func"],
            "\n-kernel: ",default_parameters["kernel"] )
        print("------------------------------------------")


        #Dummy Minimize
        if( minimizer == dummy_minimize ):
            if( save_path != None ):
                save_name = save_path + save_name 

            if(not save and not early_stop):
                for i in range( different_iteration ):
                    res.append( dummy_minimize(f, 
                                            bounds, 
                                            n_calls=number_of_call, 
                                            x0=x0[i], 
                                            y0=y0[i], 
                                            random_state=random_state,
                                            verbose= verbose,
                                            model_queue_size=model_queue_size ) )
                                            
                if plot:
                    name = plot_name + ".png"
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )
            
            elif ( ( save_step >= number_of_call and save) and  ( early_step >= number_of_call or not early_stop) ):
                for i in range( different_iteration ):
                    save_name_t = save_name + str(i) + ".pkl"
                    checkpoint_saver[i] = CheckpointSaver( save_name_t ) #save

                    res.append( dummy_minimize(f, 
                                            bounds, 
                                            n_calls=number_of_call, 
                                            x0=x0[i], 
                                            y0=y0[i], 
                                            random_state=random_state,
                                            callback=[checkpoint_saver[i] ],
                                            verbose= verbose,
                                            model_queue_size=model_queue_size ) )
                
                if plot:
                    name = plot_name + ".png"
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )
            
            elif ( save and not early_stop):

                time_eval = []

                time_t = []
                for i in range( different_iteration ):
                    save_name_t = save_name + str(i) + ".pkl"
                    checkpoint_saver[i] = CheckpointSaver( save_name_t ) #save

                    start_time = time.time()

                    res.append( dummy_minimize(f, 
                                            bounds, 
                                            n_calls=save_step, 
                                            x0=x0[i], 
                                            y0=y0[i], 
                                            random_state=random_state,
                                            callback=[checkpoint_saver[i] ],
                                            verbose= verbose,
                                            model_queue_size=model_queue_size ) )

                    end_time = time.time()
                    total_time = end_time - start_time
                    time_t.append(total_time)

                time_eval.append(time_t)

                save_csv(name_csv = save_name + ".csv",
                        dataset_name = self.dataset.get_metadata()["info"]["name"] , 
                        hyperparameters_name = self.hyperparameters, 
                        num_topic = self.model.hyperparameters['num_topics'], 
                        Surrogate = minimizer_stringa,
                        Acquisition = acq_func,
                        Time = time_eval, 
                        res = res,
                        Maximize = (self.optimization_type == 'Maximize'),
                        time_x0 = time_x0 )

                if plot:
                    name = plot_name + ".png"
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

                number_of_call_r = number_of_call - save_step

                time_t = []
                while ( number_of_call_r > 0 ) :
                    if( number_of_call_r >= save_step ):
                        for i in range( different_iteration ):
                            save_name_t = save_name + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals
                            save_name_t = "./" + save_name + str(i) + ".pkl"
                            checkpoint_saver_t = CheckpointSaver( save_name_t ) #save

                            start_time = time.time()
                            
                            res[i] = dummy_minimize(f, 
                                                bounds, 
                                                n_calls=save_step, 
                                                x0=x0_restored, 
                                                y0=y0_restored,
                                                callback=[checkpoint_saver[i] ], 
                                                random_state=random_state,
                                                verbose= verbose,
                                                model_queue_size=model_queue_size)

                            checkpoint_saver[i] = checkpoint_saver_t

                            end_time = time.time()
                            total_time = end_time - start_time
                            time_t.append(total_time)

                        time_eval.append(time_t)

                        save_csv(name_csv = save_name + ".csv",
                        dataset_name = self.dataset.get_metadata()["info"]["name"] , 
                        hyperparameters_name = self.hyperparameters, 
                        num_topic = self.model.hyperparameters['num_topics'], 
                        Surrogate = minimizer_stringa,
                        Acquisition = acq_func,
                        Time = time_eval, 
                        res = res,
                        Maximize = (self.optimization_type == 'Maximize'),
                        time_x0 = time_x0  )


                        if plot:
                            name = plot_name + ".png"
                            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )
                        number_of_call_r = number_of_call_r - save_step

                    else:
                        for i in range( different_iteration ):
                            save_name_t = save_name + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals

                            start_time = time.time()

                            res[i] = dummy_minimize(f, bounds, n_calls=number_of_call_r,
                                                    x0=x0_restored, y0=y0_restored,
                                                    callback=[checkpoint_saver[i]],
                                                    random_state=random_state,
                                                    verbose= verbose,
                                                    model_queue_size=model_queue_size)

                            end_time = time.time()
                            total_time = end_time - start_time
                            time_t.append(total_time)

                        time_eval.append(time_t)

                        save_csv(name_csv = save_name + ".csv",
                                 dataset_name = self.dataset.metadata.name ,
                                hyperparameters_name = self.hyperparameters,
                                num_topic = self.model.hyperparameters['num_topics'],
                                Surrogate = minimizer_stringa, Acquisition = acq_func,
                                Time=time_eval, res=res,
                        Maximize=(self.optimization_type == 'Maximize'),
                        time_x0=time_x0 )

                        if plot:
                            name = plot_name + ".png"
                            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )
                        number_of_call_r = number_of_call_r - save_step

            elif (not save and early_stop):

                for i in range( different_iteration ):

                    res_temp=dummy_minimize(f,
                                            bounds, 
                                            n_calls=number_of_call, 
                                            x0=x0[i], 
                                            y0=y0[i],
                                            callback= [ MyCustomEarlyStopper(
                                                        n_stop=early_step,
                                                        n_random_starts=default_parameters["n_random_starts"] )
                                                    ], 
                                            random_state=random_state,
                                            verbose= verbose,
                                            model_queue_size=model_queue_size )
                    res.append( res_temp )

                

                if plot:
                    name = plot_name + ".png"
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path=save_path )
    
            elif save and early_stop:

                for i in range( different_iteration ):
                    save_name_t = save_name + str(i) + ".pkl"
                    checkpoint_saver[i] = CheckpointSaver( save_name_t ) #save

                    res_temp = dummy_minimize(f, 
                                            bounds, 
                                            n_calls=save_step, 
                                            x0=x0[i], 
                                            y0=y0[i], 
                                            random_state=random_state,
                                            callback=[checkpoint_saver[i], 
                                                    MyCustomEarlyStopper(
                                                                n_stop=early_step,
                                                                n_random_starts=default_parameters["n_random_starts"] )],
                                            verbose= verbose,
                                            model_queue_size=model_queue_size )

                    res.append( res_temp )


                if plot:
                    name = plot_name + ".png"
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path=save_path )

                number_of_call_r = number_of_call - save_step

                while ( number_of_call_r > 0 ) :
                    
                    if( number_of_call_r >= save_step ):
                        for i in range( different_iteration ):
                            save_name_t = save_name + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals
                            save_name_t = "./" + save_name + str(i) + ".pkl"
                            checkpoint_saver_t = CheckpointSaver( save_name_t ) #save

                            res[i] = dummy_minimize(f, 
                                                bounds, 
                                                n_calls=save_step, 
                                                x0=x0_restored, 
                                                y0=y0_restored,
                                                callback=[checkpoint_saver[i], 
                                                        MyCustomEarlyStopper(
                                                                n_stop=early_step,
                                                                n_random_starts=default_parameters["n_random_starts"] ) ],
                                                random_state=random_state,
                                                verbose= verbose,
                                                model_queue_size=model_queue_size)

                            checkpoint_saver[i] = checkpoint_saver_t

                        number_of_call_r = number_of_call_r - save_step


                        if plot:
                            name = plot_name + ".png"
                            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )
                        

                    else:
                        for i in range( different_iteration ):
                            save_name_t = save_name + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals

                            res[i] = dummy_minimize(f, bounds, n_calls=number_of_call_r,
                                                    x0=x0_restored, y0=y0_restored,
                                                    callback=[checkpoint_saver[i],
                                                              MyCustomEarlyStopper(
                                                                  n_stop=early_step,
                                                                  n_random_starts=
                                                                  default_parameters["n_random_starts"])],
                                                    random_state=random_state,
                                                    verbose= verbose,
                                                    model_queue_size=model_queue_size)

                        number_of_call_r = number_of_call_r - save_step

                        if plot:
                            name = plot_name + ".png"
                            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

                if plot:
                    name = plot_name + ".png"
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

            else:
                print("Not implemented \n")

        #Forest Minimize
        if( minimizer == forest_minimize ):
            if( save_path != None ):
                save_name = save_path + save_name 

            if(not save and not early_stop):
                for i in range( different_iteration ):
                    res.append( forest_minimize(f, 
                                                bounds,
                                                base_estimator=base_estimator_forest,
                                                n_calls=number_of_call,
                                                acq_func=acq_func,
                                                n_random_starts = n_random_starts,
                                                x0=x0[i],
                                                y0=y0[i],
                                                random_state=random_state,
                                                verbose=verbose, 
                                                n_points=n_points, 
                                                xi=xi, 
                                                kappa=kappa, 
                                                n_jobs=n_jobs, 
                                                model_queue_size=model_queue_size ) )
                if plot:
                    name = plot_name + ".png"
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

            elif ( ( save_step >= number_of_call and save) and  ( early_step >= number_of_call or not early_stop ) ):
                for i in range( different_iteration ):
                    save_name_t = save_name + str(i) + ".pkl"
                    checkpoint_saver[i] = CheckpointSaver( save_name_t ) #save

                    res.append( forest_minimize(f, 
                                                bounds,
                                                base_estimator=base_estimator_forest,
                                                n_calls=number_of_call,
                                                acq_func=acq_func,
                                                n_random_starts = n_random_starts,
                                                x0=x0[i],
                                                y0=y0[i],
                                                random_state=random_state,
                                                callback=[checkpoint_saver[i] ],
                                                verbose=verbose, 
                                                n_points=n_points, 
                                                xi=xi, 
                                                kappa=kappa, 
                                                n_jobs=n_jobs, 
                                                model_queue_size=model_queue_size ) )
                if plot:
                    name = plot_name + ".png"
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )
            
            elif (save and not early_stop):

                time_eval = []

                time_t = []
                for i in range( different_iteration ):
                    save_name_t = save_name + str(i) + ".pkl"
                    checkpoint_saver[i] = CheckpointSaver( save_name_t ) #save
                    if( x0[i] == None ):
                        len_x0 = 0
                    else:
                        len_x0 = len( x0[i] )

                    flag = False
                    if( save_step >= n_random_starts + len_x0 ):
                        start_time = time.time()

                        res.append( forest_minimize(f, 
                                                bounds,
                                                base_estimator=base_estimator_forest,
                                                n_calls=save_step,
                                                acq_func=acq_func,
                                                n_random_starts = n_random_starts,
                                                x0=x0[i],
                                                y0=y0[i],
                                                random_state=random_state,
                                                callback=[checkpoint_saver[i] ],
                                                verbose=verbose, 
                                                n_points=n_points, 
                                                xi=xi, 
                                                kappa=kappa, 
                                                n_jobs=n_jobs, 
                                                model_queue_size=model_queue_size ) )
                        
                        #number_of_call_r = number_of_call - save_step

                        end_time = time.time()
                        total_time = end_time - start_time
                        time_t.append(total_time)
                    else:
                        flag = True
                        start_time = time.time()

                        res.append( forest_minimize(f, bounds,
                                                base_estimator=base_estimator_forest,
                                                n_calls=save_step + n_random_starts,
                                                acq_func=acq_func,
                                                n_random_starts = n_random_starts,
                                                x0=x0[i], y0=y0[i],
                                                random_state=random_state,
                                                callback=[checkpoint_saver[i] ],
                                                verbose=verbose, 
                                                n_points=n_points, 
                                                xi=xi, 
                                                kappa=kappa, 
                                                model_queue_size=model_queue_size ) )
                        
                        #number_of_call_r = number_of_call - save_step - n_random_starts
                        end_time = time.time()
                        total_time = end_time - start_time
                        time_t.append(total_time)

                if plot:
                    name = plot_name + ".png"
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path=save_path )

                number_of_call_r = number_of_call - save_step
                if flag:
                    fract = save_step + n_random_starts
                else:
                    fract = number_of_call - number_of_call_r   

                time_t = [i/fract for i in time_t]
                
                for i in range(fract):
                    time_eval.append(time_t)

                save_csv(name_csv = save_name + ".csv",
                        dataset_name = self.dataset.get_metadata()["info"]["name"] , 
                        hyperparameters_name = self.hyperparameters, 
                        num_topic = self.model.hyperparameters['num_topics'], 
                        Surrogate = minimizer_stringa,
                        Acquisition = acq_func,
                        Time = time_eval, 
                        res = res,
                        Maximize = (self.optimization_type == 'Maximize'),
                        time_x0 = time_x0 )


                time_t = []
                while ( number_of_call_r > 0 ) :
                    if( number_of_call_r >= save_step ):
                        for i in range( different_iteration ):
                            save_name_t = save_name + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals
                            save_name_t = "./" + save_name + str(i) + ".pkl"
                            checkpoint_saver_t = CheckpointSaver( save_name_t ) #save
                            
                            start_time = time.time()

                            res[i] = forest_minimize(f, 
                                                    bounds,
                                                    base_estimator=base_estimator_forest,
                                                    n_calls=save_step,
                                                    acq_func=acq_func,
                                                    n_random_starts = 0,
                                                    x0=x0_restored, 
                                                    y0=y0_restored,
                                                    random_state=random_state,
                                                    callback=[checkpoint_saver[i] ],
                                                    verbose=verbose, 
                                                    n_points=n_points, 
                                                    xi=xi, 
                                                    kappa=kappa, 
                                                    n_jobs=n_jobs, 
                                                    model_queue_size=model_queue_size )

                

                            checkpoint_saver[i] = checkpoint_saver_t

                            end_time = time.time()
                            total_time = end_time - start_time
                            time_t.append(total_time)

                        time_eval.append(time_t)

                        save_csv(name_csv = save_name + ".csv",
                        dataset_name = self.dataset.get_metadata()["info"]["name"] , 
                        hyperparameters_name = self.hyperparameters, 
                        num_topic = self.model.hyperparameters['num_topics'], 
                        Surrogate = minimizer_stringa,
                        Acquisition = acq_func,
                        Time = time_eval, 
                        res = res,
                        Maximize = (self.optimization_type == 'Maximize'),
                        time_x0 = time_x0  )

                        
                        if plot:
                            name = plot_name + ".png"
                            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

                        number_of_call_r = number_of_call_r - save_step

                    else:
                        for i in range( different_iteration ):
                            save_name_t = save_name + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals

                            start_time = time.time()

                            res[i] = forest_minimize(f, 
                                                    bounds,
                                                    base_estimator=base_estimator_forest,
                                                    n_calls=number_of_call_r,
                                                    acq_func=acq_func,
                                                    n_random_starts = 0,
                                                    x0=x0_restored, 
                                                    y0=y0_restored,
                                                    random_state=random_state,
                                                    callback=[checkpoint_saver[i] ],
                                                    verbose=verbose, 
                                                    n_points=n_points, 
                                                    xi=xi, 
                                                    kappa=kappa, 
                                                    n_jobs=n_jobs, 
                                                    model_queue_size=model_queue_size )

                            end_time = time.time()
                            total_time = end_time - start_time
                            time_t.append(total_time)

                        time_eval.append(time_t)

                        save_csv(name_csv = save_name + ".csv",
                        dataset_name = self.dataset.metadata.name , 
                        hyperparameters_name = self.hyperparameters, 
                        num_topic = self.model.hyperparameters['num_topics'],
                        Surrogate = minimizer_stringa,
                        Acquisition = acq_func,
                        Time = time_eval, 
                        res = res,
                        Maximize = (self.optimization_type == 'Maximize'),
                        time_x0 = time_x0 )

                        if plot:
                            name = plot_name + ".png"
                            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )
                        number_of_call_r = number_of_call_r - save_step

            elif not save and early_stop:
                for i in range(different_iteration):
                    
                    res_temp = forest_minimize(f, 
                                                bounds,
                                                base_estimator=base_estimator_forest,
                                                n_calls=number_of_call,
                                                acq_func=acq_func,
                                                n_random_starts = n_random_starts,
                                                x0=x0[i], 
                                                y0=y0[i],
                                                random_state=random_state,
                                                callback=[ MyCustomEarlyStopper(
                                                                n_stop = early_step,
                                                                n_random_starts = default_parameters["n_random_starts"] ) ],
                                                verbose=verbose, 
                                                n_points=n_points, 
                                                xi=xi, 
                                                kappa=kappa, 
                                                n_jobs=n_jobs, 
                                                model_queue_size=model_queue_size )

                    res.append( res_temp )

                

                if plot:
                    name = plot_name + ".png"
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )
    
            elif save and early_stop:
                for i in range( different_iteration ):
                    save_name_t = save_name + str(i) + ".pkl"
                    checkpoint_saver[i] = CheckpointSaver(save_name_t) #save

                    if x0[i] == None:
                        len_x0 = 0
                    else:
                        len_x0 = len(x0[i])

                    if save_step >= n_random_starts + len_x0:
                        res_temp = forest_minimize(f, 
                                                    bounds,
                                                    base_estimator=base_estimator_forest,
                                                    n_calls=save_step,
                                                    acq_func=acq_func,
                                                    n_random_starts = n_random_starts,
                                                    x0=x0[i], 
                                                    y0=y0[i],
                                                    random_state=random_state,
                                                    callback=[checkpoint_saver[i], 
                                                            MyCustomEarlyStopper(
                                                                    n_stop = early_step,
                                                                    n_random_starts=default_parameters["n_random_starts"] ) ],
                                                    verbose=verbose, 
                                                    n_points=n_points, 
                                                    xi=xi, 
                                                    kappa=kappa, 
                                                    n_jobs=n_jobs, 
                                                    model_queue_size=model_queue_size )
                    else:
                        res_temp = forest_minimize(f, 
                                                bounds,
                                                base_estimator=base_estimator_forest,
                                                n_calls=save_step + n_random_starts,
                                                acq_func=acq_func,
                                                n_random_starts = n_random_starts,
                                                x0=x0[i], 
                                                y0=y0[i],
                                                random_state=random_state,
                                                callback=[checkpoint_saver[i], 
                                                        MyCustomEarlyStopper(
                                                                n_stop = early_step,
                                                                n_random_starts = default_parameters["n_random_starts"] ) ], 
                                                verbose=verbose, 
                                                n_points=n_points, 
                                                xi=xi, 
                                                kappa=kappa, 
                                                model_queue_size=model_queue_size )

                    res.append( res_temp )

                if plot:
                    name = plot_name + ".png"
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

                number_of_call_r = number_of_call - save_step

                while ( number_of_call_r > 0 ) :
                    
                    if( number_of_call_r >= save_step ):
                        for i in range( different_iteration ):
                            save_name_t = save_name + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals
                            save_name_t = "./" + save_name + str(i) + ".pkl"
                            checkpoint_saver_t = CheckpointSaver( save_name_t ) #save

                            if( x0[i] == None ):
                                len_x0 = 0
                            else:
                                len_x0 = len( x0[i] )

                            if( save_step >= len_x0 ):
                                res[i] = forest_minimize(f, 
                                                    bounds,
                                                    base_estimator=base_estimator_forest,
                                                    n_calls=save_step,
                                                    acq_func=acq_func,
                                                    n_random_starts = 0,
                                                    x0=x0_restored, 
                                                    y0=y0_restored,
                                                    random_state=random_state,
                                                    callback=[checkpoint_saver[i],
                                                            MyCustomEarlyStopper(
                                                                        n_stop = early_step,
                                                                        n_random_starts = default_parameters["n_random_starts"] ) ], 
                                                    verbose=verbose, 
                                                    n_points=n_points, 
                                                    xi=xi, 
                                                    kappa=kappa, 
                                                    n_jobs=n_jobs, 
                                                    model_queue_size=model_queue_size )
                            else:
                                res[i] = forest_minimize(f, 
                                                bounds,
                                                base_estimator=base_estimator_forest,
                                                n_calls=save_step + len_x0,
                                                acq_func=acq_func,
                                                n_random_starts = 0,
                                                x0=x0_restored, 
                                                y0=y0_restored,
                                                random_state=random_state,
                                                callback=[checkpoint_saver[i],
                                                        MyCustomEarlyStopper(
                                                                n_stop = early_step,
                                                                n_random_starts = default_parameters["n_random_starts"] ) ], 
                                                verbose=verbose, 
                                                n_points=n_points, 
                                                xi=xi, 
                                                kappa=kappa, 
                                                model_queue_size=model_queue_size )

                            checkpoint_saver[i] = checkpoint_saver_t

                        number_of_call_r = number_of_call_r - save_step

                        if plot:
                            name = plot_name + ".png"
                            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

                    else:
                        for i in range( different_iteration ):
                            save_name_t = save_name + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals

                            if( x0[i] == None ):
                                len_x0 = 0
                            else:
                                len_x0 = len( x0[i] )


                            if( save_step >= n_random_starts + len_x0 ):
                                res[i] = forest_minimize(f, 
                                                        bounds,
                                                        base_estimator=base_estimator_forest,
                                                        n_calls=number_of_call_r,
                                                        acq_func=acq_func,
                                                        n_random_starts = 0,
                                                        x0=x0_restored,
                                                        y0=y0_restored,
                                                        random_state=random_state,
                                                        callback=[checkpoint_saver[i], 
                                                                MyCustomEarlyStopper(
                                                                        n_stop = early_step,
                                                                        n_random_starts = default_parameters["n_random_starts"] ) ], 
                                                        verbose=verbose, 
                                                        n_points=n_points, 
                                                        xi=xi, 
                                                        kappa=kappa, 
                                                        model_queue_size=model_queue_size )


                            else:
                                res[i] = forest_minimize(f, 
                                                        bounds,
                                                        base_estimator=base_estimator_forest,
                                                        n_calls=number_of_call_r + len_x0,
                                                        acq_func=acq_func,
                                                        n_random_starts = 0,
                                                        x0=x0_restored,
                                                        y0=y0_restored,
                                                        random_state=random_state,
                                                        callback=[checkpoint_saver[i], 
                                                                MyCustomEarlyStopper(
                                                                        n_stop = early_step,
                                                                        n_random_starts = default_parameters["n_random_starts"] ) ], 
                                                        verbose=verbose, 
                                                        n_points=n_points, 
                                                        xi=xi, 
                                                        kappa=kappa, 
                                                        model_queue_size=model_queue_size ) 

                        number_of_call_r = number_of_call_r - save_step

                        if plot:
                            name = plot_name + ".png"
                            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )
    
        
                if plot:
                    name = plot_name + ".png"
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

            else:
                print("Not implemented \n")
                
        #GP Minimize
        if( minimizer == gp_minimize ):
            if(not save and not early_stop ):
                for i in range( different_iteration ):
                    gpr = GaussianProcessRegressor(kernel=kernel, 
                                                alpha=alpha,
                                                normalize_y=True, 
                                                noise="gaussian",
                                                n_restarts_optimizer=0,
                                                random_state = random_state)

                    opt = skopt_optimizer(bounds, 
                                    base_estimator=gpr, 
                                    acq_func=acq_func,
                                    n_random_starts = n_random_starts,
                                    n_initial_points= n_random_starts,
                                    acq_optimizer="sampling", 
                                    random_state=random_state,
                                    model_queue_size=model_queue_size )

                    if( x0[i] != None and y0[i] != None):
                        opt.tell(x0[i], y0[i], fit=True)
                    res.append( opt.run(f, number_of_call) )

                if plot:
                    name = plot_name + ".png"
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path=save_path )

            elif ( ( save_step >= number_of_call and save) and  ( early_step >= number_of_call or not early_stop )  ):
                for i in range( different_iteration ):
                    
                    gpr = GaussianProcessRegressor(kernel=kernel, 
                                                alpha=alpha,
                                                normalize_y=True, 
                                                noise="gaussian",
                                                n_restarts_optimizer=0,
                                                random_state = random_state)

                    opt = skopt_optimizer(bounds, 
                                    base_estimator=gpr, 
                                    acq_func=acq_func,
                                    n_random_starts = n_random_starts,
                                    n_initial_points= n_random_starts,
                                    acq_optimizer="sampling", 
                                    random_state=random_state,
                                    model_queue_size=model_queue_size )

                    if( x0[i] != None and y0[i] != None):
                        opt.tell(x0[i], y0[i], fit=True)

                    res_t = opt.run(f, number_of_call)
                    res.append( res_t )

                checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save

                if plot:
                    name = plot_name + ".png"
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

            elif save and not early_stop:

                time_eval = []

                time_t = []
                for i in range( different_iteration ):

                    start_time = time.time()

                    gpr = GaussianProcessRegressor(kernel=kernel, 
                                                alpha=alpha,
                                                normalize_y=True, 
                                                noise="gaussian",
                                                n_restarts_optimizer=0,
                                                random_state = random_state)

                    opt = skopt_optimizer(bounds, 
                                    base_estimator=gpr, 
                                    acq_func=acq_func,
                                    n_random_starts = n_random_starts,
                                    n_initial_points= n_random_starts,
                                    acq_optimizer="sampling", 
                                    random_state=random_state,
                                    model_queue_size=model_queue_size )

                    if( x0[i] != None and y0[i] != None):
                        opt.tell(x0[i], y0[i], fit=True)

                    res_t = opt.run(f, save_step)
                    res.append( res_t )

                    end_time = time.time()
                    total_time = end_time - start_time
                    time_t.append(total_time)

                time_eval.append(time_t)

                save_csv(name_csv = save_path + save_name + ".csv",
                        dataset_name = self.dataset.get_metadata()["info"]["name"] , 
                        hyperparameters_name = self.hyperparameters, 
                        num_topic = self.model.hyperparameters['num_topics'], 
                        Surrogate = minimizer_stringa,
                        Acquisition = acq_func,
                        Time = time_eval, 
                        res = res,
                        Maximize = (self.optimization_type == 'Maximize'),
                        time_x0 = time_x0 )


                checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                number_of_call_r = number_of_call - save_step

                if plot:
                    name = plot_name + ".png"
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )
                

                time_t = []
                while ( number_of_call_r > 0 ) :
                    if( number_of_call_r >= save_step ):
                        partial_res = tool.load_BO( checkpoint_saver ) #restore

                        for i in range( different_iteration ):
                            x0_restored = partial_res[i].x_iters
                            y0_restored = list(partial_res[i].func_vals)

                            start_time = time.time()

                            gpr = GaussianProcessRegressor(kernel=kernel, 
                                                        alpha=alpha,
                                                        normalize_y=True, 
                                                        noise="gaussian",
                                                        n_restarts_optimizer=0,
                                                        random_state = random_state)

                            opt = skopt_optimizer(bounds, 
                                    base_estimator=gpr, 
                                    acq_func=acq_func,
                                    n_random_starts = 0,
                                    n_initial_points= 0,
                                    acq_optimizer="sampling", 
                                    random_state=random_state,
                                    model_queue_size=model_queue_size )

                            opt.tell(x0_restored, y0_restored, fit=True)

                            res_t = opt.run(f, save_step)
                            res[i] = res_t

                            end_time = time.time()
                            total_time = end_time - start_time
                            time_t.append(total_time)

                        time_eval.append(time_t)

                        save_csv(name_csv = save_path + save_name + ".csv",
                        dataset_name = self.dataset.get_metadata()["info"]["name"] , 
                        hyperparameters_name = self.hyperparameters, 
                        num_topic = self.model.hyperparameters['num_topics'], 
                        Surrogate = minimizer_stringa,
                        Acquisition = acq_func,
                        Time = time_eval, 
                        res = res,
                        Maximize = (self.optimization_type == 'Maximize'),
                        time_x0 = time_x0 )


                        checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                        number_of_call_r = number_of_call_r - save_step

                        if plot:
                            name = plot_name + ".png"
                            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

                    else:
                        partial_res = tool.load_BO( checkpoint_saver ) #restore
                        for i in range( different_iteration ):
                            x0_restored = partial_res[i].x_iters
                            y0_restored = list(partial_res[i].func_vals)

                            start_time = time.time()

                            gpr = GaussianProcessRegressor(kernel=kernel, 
                                                        alpha=alpha,
                                                        normalize_y=True, 
                                                        noise="gaussian",
                                                        n_restarts_optimizer=0,
                                                        random_state = random_state)

                            opt = skopt_optimizer(bounds, 
                                    base_estimator=gpr, 
                                    acq_func=acq_func,
                                    n_random_starts = 0,
                                    n_initial_points= 0,
                                    acq_optimizer="sampling", 
                                    random_state=random_state,
                                    model_queue_size=model_queue_size )

                            opt.tell(x0_restored, y0_restored, fit=True)

                            res_t = opt.run(f, number_of_call_r)
                            res[i] = res_t

                            end_time = time.time()
                            total_time = end_time - start_time
                            time_t.append(total_time)

                        time_eval.append(time_t)

                        save_csv(name_csv = save_path + save_name + ".csv",
                        dataset_name = self.dataset.get_metadata()["info"]["name"] , 
                        hyperparameters_name = self.hyperparameters, 
                        num_topic = self.model.hyperparameters['num_topics'], 
                        Surrogate = minimizer_stringa,
                        Acquisition = acq_func,
                        Time = time_eval, 
                        res = res,
                        Maximize = (self.optimization_type == 'Maximize'),
                        time_x0 = time_x0  )


                        checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                        number_of_call_r = number_of_call_r - save_step

                        if plot:
                            name = plot_name + ".png"
                            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

            elif (not save and early_stop):

                early_stop_flag = [False] * different_iteration
                
                for i in range( different_iteration ):
                    if( early_stop_flag[i] == False ):
                        gpr = GaussianProcessRegressor(kernel=kernel, 
                                                    alpha=alpha,
                                                    normalize_y=True, 
                                                    noise="gaussian",
                                                    n_restarts_optimizer=0,
                                                    random_state = random_state)

                        opt = skopt_optimizer(bounds, 
                                        base_estimator=gpr, 
                                        acq_func=acq_func,
                                        n_random_starts = n_random_starts,
                                        n_initial_points= n_random_starts,
                                        acq_optimizer="sampling", 
                                        random_state=random_state,
                                        model_queue_size=model_queue_size )

                        if( x0[i] != None and y0[i] != None):
                            opt.tell(x0[i], y0[i], fit=True)

                        res_t = opt.run(f, early_step)
                        if tool.early_condition(res_t, early_step, n_random_starts):
                           early_stop_flag[i] = True

                        res.append( res_t )

                

                checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                number_of_call_r = number_of_call - early_step

                if plot:
                    name = plot_name + ".png"
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )
                

                while ( number_of_call_r > 0 ) :
                    if( number_of_call_r >= early_step ):
                        partial_res = tool.load_BO( checkpoint_saver ) #restore

                        for i in range( different_iteration ):
                            if( early_stop_flag[i] == False ):
                                x0_restored = partial_res[i].x_iters
                                y0_restored = list(partial_res[i].func_vals)

                                gpr = GaussianProcessRegressor(kernel=kernel, 
                                                            alpha=alpha,
                                                            normalize_y=True, 
                                                            noise="gaussian",
                                                            n_restarts_optimizer=0,
                                                            random_state = random_state)

                                opt = skopt_optimizer(bounds, 
                                        base_estimator=gpr, 
                                        acq_func=acq_func,
                                        n_random_starts = 0,
                                        n_initial_points= 0,
                                        acq_optimizer="sampling", 
                                        random_state=random_state,
                                        model_queue_size=model_queue_size )

                                opt.tell(x0_restored, y0_restored, fit=True)

                                res_t = opt.run(f, early_step)
                                if tool.early_condition(res_t, early_step, n_random_starts):
                                    early_stop_flag[i] = True

                                res[i] = res_t

                        checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                        number_of_call_r = number_of_call_r - early_step

                        if plot:
                            name = plot_name + ".png"
                            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

                    else:
                        partial_res = tool.load_BO( checkpoint_saver ) #restore
                        for i in range( different_iteration ):
                            if( early_stop_flag[i] == False ):
                                x0_restored = partial_res[i].x_iters
                                y0_restored = list(partial_res[i].func_vals)

                                gpr = GaussianProcessRegressor(kernel=kernel, 
                                                            alpha=alpha,
                                                            normalize_y=True, 
                                                            noise="gaussian",
                                                            n_restarts_optimizer=0,
                                                            random_state = random_state)
    
                                opt = skopt_optimizer(bounds, 
                                        base_estimator=gpr, 
                                        acq_func=acq_func,
                                        n_random_starts = 0,
                                        n_initial_points= 0,
                                        acq_optimizer="sampling", 
                                        random_state=random_state,
                                        model_queue_size=model_queue_size )

                                opt.tell(x0_restored, y0_restored, fit=True)

                                res_t = opt.run(f, number_of_call_r)
                                if tool.early_condition(res_t, early_step, n_random_starts):
                                    early_stop_flag[i] = True
                                    
                                res[i] = res_t

                        checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                        number_of_call_r = number_of_call_r - early_step

                        if plot:
                            name = plot_name + ".png"
                            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )
    
            elif ( save and early_stop):

                early_stop_flag = [False] * different_iteration
                
                for i in range( different_iteration ):
                    if( early_stop_flag[i] == False ):
                        gpr = GaussianProcessRegressor(kernel=kernel, 
                                                    alpha=alpha,
                                                    normalize_y=True, 
                                                    noise="gaussian",
                                                    n_restarts_optimizer=0,
                                                    random_state = random_state)

                        opt = skopt_optimizer(bounds, 
                                        base_estimator=gpr, 
                                        acq_func=acq_func,
                                        n_random_starts = n_random_starts,
                                        n_initial_points= n_random_starts,
                                        acq_optimizer="sampling", 
                                        random_state=random_state,
                                        model_queue_size=model_queue_size )

                        if( x0[i] != None and y0[i] != None):
                            opt.tell(x0[i], y0[i], fit=True)

                        res_t = opt.run(f, save_step)
                        if tool.early_condition(res_t, early_step, n_random_starts):
                           early_stop_flag[i] = True

                        res.append( res_t )

                

                checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                number_of_call_r = number_of_call - save_step

                if plot:
                    name = plot_name + ".png"
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )
                

                while ( number_of_call_r > 0 ) :
                    if( number_of_call_r >= save_step ):
                        partial_res = tool.load_BO( checkpoint_saver ) #restore

                        for i in range( different_iteration ):
                            if( early_stop_flag[i] == False ):
                                x0_restored = partial_res[i].x_iters
                                y0_restored = list(partial_res[i].func_vals)

                                gpr = GaussianProcessRegressor(kernel=kernel, 
                                                            alpha=alpha,
                                                            normalize_y=True, 
                                                            noise="gaussian",
                                                            n_restarts_optimizer=0,
                                                            random_state = random_state)

                                opt = skopt_optimizer(bounds, 
                                        base_estimator=gpr, 
                                        acq_func=acq_func,
                                        n_random_starts = 0,
                                        n_initial_points= 0,
                                        acq_optimizer="sampling", 
                                        random_state=random_state,
                                        model_queue_size=model_queue_size )

                                opt.tell(x0_restored, y0_restored, fit=True)

                                res_t = opt.run(f, save_step)
                                if tool.early_condition(res_t, early_step, n_random_starts):
                                    early_stop_flag[i] = True

                                res[i] = res_t

                        checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                        number_of_call_r = number_of_call_r - save_step

                        if plot:
                            name = plot_name + ".png"
                            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

                    else:
                        partial_res = tool.load_BO( checkpoint_saver ) #restore
                        for i in range( different_iteration ):
                            if( early_stop_flag[i] == False ):
                                x0_restored = partial_res[i].x_iters
                                y0_restored = list(partial_res[i].func_vals)

                                gpr = GaussianProcessRegressor(kernel=kernel, 
                                                            alpha=alpha,
                                                            normalize_y=True, 
                                                            noise="gaussian",
                                                            n_restarts_optimizer=0,
                                                            random_state = random_state)
    
                                opt = skopt_optimizer(bounds, 
                                        base_estimator=gpr, 
                                        acq_func=acq_func,
                                        n_random_starts = 0,
                                        n_initial_points= 0,
                                        acq_optimizer="sampling", 
                                        random_state=random_state,
                                        model_queue_size=model_queue_size )

                                opt.tell(x0_restored, y0_restored, fit=True)

                                res_t = opt.run(f, number_of_call_r)
                                if tool.early_condition(res_t, early_step, n_random_starts):
                                    early_stop_flag[i] = True
                                    
                                res[i] = res_t

                        checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                        number_of_call_r = number_of_call_r - save_step

                        if plot:
                            name = plot_name + ".png"
                            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

            else:
                print("Not implemented \n")

        return res

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
        #print("default parameters ", default_parameters )
        self.extra_metrics = default_parameters["extra_metrics"]

        self.optimization_type = default_parameters['optimization_type']

        # Optimization call
        optimize_result = self.Bayesian_optimization(
                            f = self._objective_function,
                            bounds = params_space_list,
                            minimizer = default_parameters["minimizer"],
                            number_of_call = default_parameters["n_calls"],
                            different_iteration = default_parameters["different_iteration"],
                            kernel = default_parameters["kernel"],
                            acq_func = default_parameters["acq_func"],
                            base_estimator_forest=default_parameters["base_estimator"],
                            random_state = default_parameters["random_state"],
                            noise_level = default_parameters["noise"],
                            alpha = default_parameters["alpha"],
                            kappa = default_parameters["kappa"],
                            X0 = default_parameters["x0"],
                            Y0 = default_parameters["y0"],
                            time_x0 = default_parameters ["time_x0"],
                            n_random_starts = default_parameters["n_random_starts"],
                            save = default_parameters["save"],
                            save_step = default_parameters["save_step"],
                            save_name = default_parameters["save_name"],
                            save_path = default_parameters["save_path"],
                            early_stop = default_parameters["early_stop"],
                            early_step = default_parameters["early_step"],
                            plot = default_parameters["plot"],
                            plot_name = default_parameters["plot_name"],
                            log_scale_plot = default_parameters["log_scale_plot"],
                            verbose = default_parameters["verbose"],
                            n_points = default_parameters["n_points"],
                            xi  = default_parameters["xi"],
                            n_jobs = default_parameters["n_jobs"],
                            model_queue_size = default_parameters["model_queue_size"]
        )    


        # To have the right result
        if self.optimization_type == 'Maximize':
            for i in range( len(optimize_result) ):
                optimize_result[i].fun = - optimize_result[i].fun
                for j in range( len(optimize_result[i].func_vals) ):
                    optimize_result[i].func_vals[j] = - optimize_result[i].func_vals[j]

            if( default_parameters["plot"] ):
                tool.plot_bayesian_optimization( list_of_res = optimize_result, 
                                    name_plot = default_parameters["plot_name"],
                                    log_scale = default_parameters["log_scale_plot"], 
                                    path = default_parameters["save_path"],
                                    conv_min = False)


        # Create Best_evaluation object from optimization results
        result = Best_evaluation(self.hyperparameters,
                                 optimize_result,
                                 self.optimization_type == 'Maximize', #Maximize = (self.optimization_type == 'Maximize')
                                 self._iterations,
                                 self.metric.__class__.__name__)

        

        return result
