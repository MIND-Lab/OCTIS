#To do:
#Eealy-stop condition
#Log scale for plot

from skopt.space.space import Real, Integer
from skopt.utils import dimensions_aslist
from optimization.optimization_result import Best_evaluation

import matplotlib.pyplot as plt
import numpy as np
import os
import math
import statistics
from scipy.spatial import distance as dist_eu
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
from skopt import dump, load
from PIL import Image
import inspect, re

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
    'verbose': False,#
    'n_points': 10000,#
    'base_estimator': 'RF',
    'kappa': 1.96,#
    'alpha': 1e-10,
    'x0': [None],
    'y0': [None],
    'xi': 1.96,#
    'n_jobs': 1,#
    'model_queue_size': None,#
    'callback': None,#
    'optimization_type': 'Maximize',
    'extra_metrics': [],
    'save': False, 
    'save_step': 1, 
    'save_name': "partial_result", 
    'early_stop': False, # Not working yet
    'early_step': 10, 
    'plot': False, 
    'plot_name': "Bayesian optimization plot" 
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
        self.optimization_parameters = optimization_parameters
        
    def _objective_function(self, hyperparameters):
        """
        objective function to optimize

        Parameters
        ----------
        hyperparameters : dictionary of hyperparameters
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

    def get_concat_h(self, im1, im2):
        """
            Concat two images as it follows:
            -    im1 = Image.open('Comparing Acquisition Function Mean.png')
            -    im2 = Image.open('Comparing Acquisition Function Mean 1x.png')
            -    get_concat_h(im1, im2).save('h.jpg')

            -PIL.Image module needed

            Parameters
            ----------
            im1 : First image

            im2 : Second image 

            Returns
            -------
            dst : im1 and im2 concatenation
                
        """    
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def varname(self, p):
        """
            Return the name of the variabile p
            -inspect module needed
            -re module needed

            Parameters
            ----------
            p : variable with a name

            Returns
            -------
            m : Name of the variable p
                
        """    
        for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
            m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
            if m:
                return m.group(1)

    def print_func_vals(self, list_of_res):
        """
            Print the function's values of a
            Bayesian_optimization result 

            Parameters
            ----------
            list_of_res : A Bayesian_optimization result

        """    
        for i in range( len(list_of_res) ):
            print( list_of_res[i].func_vals )

    def print_x_iters(self, list_of_res):
        """
            Print the x iteration of a
            Bayesian_optimization result 

            Parameters
            ----------
            list_of_res : A Bayesian_optimization result

        """    
        for i in range( len(list_of_res) ):
            print( list_of_res[i].x_iters )

    def random_generator(self, bounds, n , n_iter):
        """
            Return a list of n random numbers in the bounds
            repeat itself for n_iter iteration.
            Random numbers are generated with 
            uniform distribution
            -np.random.uniform module needed

            Parameters
            ----------
            bounds : A list of bound for the random numbers

            n : Number of random numbers for each iteration

            n_iter : Number of iterations

            Returns
            -------
            array : A list of n*n_iter random numbers 
                    in the bounds
                
        """    
        array = []
        for i in range( n_iter ):
            array.append( [] )
        for i in range( n_iter ):
            for j in range( n ):
                array[i].append( [] )
        dimensione = len( bounds )
        for i in range( n_iter ):
            for j in range( n ):
                for d in range( dimensione ):
                    array[i][j].append( np.random.uniform(low = bounds[d][0], 
                                                        high = bounds[d][1]) )
        return array

    def funct_eval(self, funct, points):
        """
            Return a list of the evaluation of the points 
            in the function funct
            Build to work with random_generator()

            Parameters
            ----------
            funct : A function the return a single value

            points : A list of point

            Returns
            -------
            array : A list of evaluation
        """    
        array = []
        for i in range( len( points ) ):
            array.append( [] )
        for i in range( len( points ) ):
            for j in range( len( points[0] ) ):
                array[i].append( funct( points[i][j] ) )
        return array

    def plot_bayesian_optimization(self, list_of_res, name_plot = "plot_BO" ):
        """
            Save a plot of the result of a Bayesian_optimization 
            considering mean and standard deviation

            Parameters
            ----------
            list_of_res : A Bayesian_optimization result

            name_plot : The name of the file you want to 
                        give to the plot

        """    
        media = self.total_mean( list_of_res )
        array = [ i for i in range( len( media ) ) ]
        plt.plot(array, media, color='blue', label= "res" )

        plt.fill_between(array, 
                        self.lower_standard_deviation( list_of_res ), 
                        self.upper_standard_deviation( list_of_res ),
                        color='blue', alpha=0.2)

        plt.ylabel('min f(x) after n calls')
        plt.xlabel('Number of calls n')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(True)
        plt.savefig( name_plot ) 
        plt.clf()

    class MyCustomEarlyStopper(EarlyStopper):
        """
            Stop the optimization if the best minima
            doesn't change for n_stop iteration
        """
        def __init__(self, n_stop=10):
            """
                Inititalize the early stopper 

                Parameters
                ----------
                n_stop : number of evaluation without improvement

            """
			#super.(EarlyStopper, self).__init__()
            EarlyStopper(EarlyStopper, self).__init__()
            self.n_stop = n_stop

        def _criterion(self, result):
            """
                Compute the decision to stop or not.

                Parameters
                ----------
                result : `OptimizeResult`, scipy object
                        The optimization as a OptimizeResult object.

                Returns
                -------
                decision : boolean or None
                        Return True/False if the criterion can make a decision or `None` if
                        there is not enough data yet to make a decision.

            """
            if len(result.func_vals) >= self.n_stop + default_parameters["n_random_starts"]:
                func_vals = super.convergence_res( result ) #not sure
                #print("func_vals", func_vals)
                worst = func_vals[ len(func_vals) - (self.n_stop) ]
                best = func_vals[-1]
                #print("diff ", worst - best )
                return worst - best == 0

            else:
                return None

    #Da commentare
    def Bayesian_optimization(self,
                            f ,#= self.self._objective_function,#
                            bounds ,#= params_space_list,#
                            minimizer = default_parameters["minimizer"],
                            number_of_call = default_parameters["n_calls"],
                            different_iteration = default_parameters["different_iteration"],
                            kernel = default_parameters["kernel"],
                            acq_func = default_parameters["acq_func"],
                            base_estimator_forest=default_parameters["base_estimator"],
                            random_state = default_parameters["random_state"],
                            noise_level = default_parameters["noise"],
                            alpha = default_parameters["alpha"],
                            X0 = default_parameters["x0"],
                            Y0 = default_parameters["y0"],
                            n_random_starts = default_parameters["n_random_starts"],
                            save = default_parameters["save"],
                            save_step = default_parameters["save_step"],
                            save_name = default_parameters["save_name"],
                            early_stop = default_parameters["early_stop"],
                            early_step = default_parameters["early_step"],
                            plot = default_parameters["plot"],
                            plot_name = default_parameters["plot_name"]
        ):
        
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

        #Dummy Minimize
        if( minimizer == dummy_minimize ):
            print("Dummy with ", number_of_call, " number of call and ", different_iteration," different test")
            if( save == False and early_stop == False ):
                for i in range( different_iteration ):
                    res.append( dummy_minimize(f, 
                                            bounds, 
                                            n_calls=number_of_call, 
                                            x0=x0[i], 
                                            y0=y0[i], 
                                            random_state=random_state) )
                                            
                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )
            
            elif ( ( save_step >= number_of_call and save == True ) and  ( early_step >= number_of_call or early_stop == False ) ):
                for i in range( different_iteration ):
                    save_name_t = save_name + "_" + str(i) + ".pkl"
                    checkpoint_saver[i] = CheckpointSaver( save_name_t ) #save

                    res.append( dummy_minimize(f, 
                                            bounds, 
                                            n_calls=number_of_call, 
                                            x0=x0[i], 
                                            y0=y0[i], 
                                            random_state=random_state,
                                            callback=[checkpoint_saver[i] ] ) )
                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )
            
            elif ( save == True and early_stop == False ):
                for i in range( different_iteration ):
                    save_name_t = save_name + "_" + str(i) + ".pkl"
                    checkpoint_saver[i] = CheckpointSaver( save_name_t ) #save

                    res.append( dummy_minimize(f, 
                                            bounds, 
                                            n_calls=save_step, 
                                            x0=x0[i], 
                                            y0=y0[i], 
                                            random_state=random_state,
                                            callback=[checkpoint_saver[i] ] ) )

                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )

                number_of_call_r = number_of_call - save_step

                while ( number_of_call_r > 0 ) :
                    if( number_of_call_r >= save_step ):
                        for i in range( different_iteration ):
                            save_name_t = save_name + "_" + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals
                            save_name_t = "./" + save_name + "_" + str(i) + ".pkl"
                            checkpoint_saver_t = CheckpointSaver( save_name_t ) #save

                            res[i] = dummy_minimize(f, 
                                                bounds, 
                                                n_calls=save_step, 
                                                x0=x0_restored, 
                                                y0=y0_restored,
                                                callback=[checkpoint_saver[i] ], 
                                                random_state=random_state)

                            checkpoint_saver[i] = checkpoint_saver_t
                        if( plot == True ):
                            name = plot_name + ".png"
                            self.plot_bayesian_optimization( res, name )
                        number_of_call_r = number_of_call_r - save_step

                    else:
                        for i in range( different_iteration ):
                            save_name_t = save_name + "_" + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals

                            res[i] = dummy_minimize(f, 
                                                bounds, 
                                                n_calls=number_of_call_r, 
                                                x0=x0_restored, 
                                                y0=y0_restored,
                                                callback=[checkpoint_saver[i] ], 
                                                random_state=random_state)

                        if( plot == True ):
                            name = plot_name + ".png"
                            self.plot_bayesian_optimization( res, name )
                        number_of_call_r = number_of_call_r - save_step

            elif ( save == False and early_stop == True ):

                for i in range( different_iteration ):

                    res_temp = dummy_minimize(f, 
                                            bounds, 
                                            n_calls=number_of_call, 
                                            x0=x0[i], 
                                            y0=y0[i],
                                            callback= [ self.MyCustomEarlyStopper( n_stop = early_step ) ], 
                                            random_state=random_state )

                    if( len( res_temp.func_vals ) < number_of_call + len( y0[i] ) ):
                        #print( "EARLY_f" )
                        lenght_complementare = number_of_call - len( res_temp.func_vals ) + len( y0[i] )
                        func_vals_temp = []
                        for i in range( len(res_temp.func_vals) ) :
                            func_vals_temp.append( res_temp.func_vals[i])
                        for i in range( lenght_complementare ) :
                            func_vals_temp.append( res_temp.func_vals[-1] )
                        res_temp.func_vals = func_vals_temp
                        res_temp.x_iters = res_temp.x_iters + ( [ res_temp.x_iters[-1] ] * ( lenght_complementare ) )
                        #print( len(res_temp.func_vals), " res_temp.func_vals ", res_temp.func_vals )
                        #print( len(res_temp.x_iters),  " res_temp.x_iters ", res_temp.x_iters )
                    res.append( res_temp )

                

                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )
    
            elif ( save == True and early_stop == True ):

                for i in range( different_iteration ):
                    save_name_t = save_name + "_" + str(i) + ".pkl"
                    checkpoint_saver[i] = CheckpointSaver( save_name_t ) #save

                    res_temp = dummy_minimize(f, 
                                            bounds, 
                                            n_calls=save_step, 
                                            x0=x0[i], 
                                            y0=y0[i], 
                                            random_state=random_state,
                                            callback=[checkpoint_saver[i], self.MyCustomEarlyStopper( n_stop = early_step ) ] )

                    if( len( res_temp.func_vals ) < save_step + len( y0[i] ) ):
                        lenght_complementare = save_step + len( y0[i] ) - len( res_temp.func_vals )
                        func_vals_temp = []
                        for j in range( len(res_temp.func_vals) ) :
                            func_vals_temp.append( res_temp.func_vals[j])
                        for j in range( lenght_complementare ) :
                            func_vals_temp.append( res_temp.func_vals[-1] )
                        res_temp.func_vals = func_vals_temp
                        res_temp.x_iters = res_temp.x_iters + ( [ res_temp.x_iters[-1] ] * ( lenght_complementare ) )
                    
                    res.append( res_temp )

                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )

                number_of_call_r = number_of_call - save_step

                while ( number_of_call_r > 0 ) :
                    
                    if( number_of_call_r >= save_step ):
                        for i in range( different_iteration ):
                            save_name_t = save_name + "_" + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals
                            save_name_t = "./" + save_name + "_" + str(i) + ".pkl"
                            checkpoint_saver_t = CheckpointSaver( save_name_t ) #save

                            res[i] = dummy_minimize(f, 
                                                bounds, 
                                                n_calls=save_step, 
                                                x0=x0_restored, 
                                                y0=y0_restored,
                                                callback=[checkpoint_saver[i], self.MyCustomEarlyStopper( n_stop = early_step ) ], 
                                                random_state=random_state)

                            checkpoint_saver[i] = checkpoint_saver_t

                        number_of_call_r = number_of_call_r - save_step

                        for i in range( different_iteration ):
                            if( len( res[i].func_vals ) < number_of_call + len( y0[i] ) - number_of_call_r ):
                                #print("before ", res[i].func_vals )
                                lenght_complementare = number_of_call + len( y0[i] ) - number_of_call_r - len( res[i].func_vals )
                                #print("Len_c ", lenght_complementare)
                                #print("Len fun", len( res[i].func_vals ))
                                func_vals_temp = []
                                for j in range( len( res[i].func_vals ) ) :
                                    func_vals_temp.append( res[i].func_vals[j])
                                for j in range( lenght_complementare ) :
                                    func_vals_temp.append( res[i].func_vals[-1] )
                                res[i].func_vals = func_vals_temp
                                res[i].x_iters = res[i].x_iters + ( [ res[i].x_iters[-1] ] * ( lenght_complementare ) )
                                #print("after ", res[i].func_vals )


                        if( plot == True ):
                            name = plot_name + ".png"
                            self.plot_bayesian_optimization( res, name )
                        

                    else:
                        for i in range( different_iteration ):
                            save_name_t = save_name + "_" + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals

                            res[i] = dummy_minimize(f, 
                                                bounds, 
                                                n_calls=number_of_call_r, 
                                                x0=x0_restored, 
                                                y0=y0_restored,
                                                callback=[checkpoint_saver[i], self.MyCustomEarlyStopper( n_stop = early_step ) ], 
                                                random_state=random_state)

                        number_of_call_r = number_of_call_r - save_step

                        for i in range( different_iteration ):
                            if( len( res[i].func_vals ) < number_of_call + len( y0[i] ) ):
                                lenght_complementare = number_of_call - len( res[i].func_vals ) + len( y0[i] )
                                func_vals_temp = []
                                for j in range( len(res_temp.func_vals) ) :
                                    func_vals_temp.append( res_temp.func_vals[j])
                                for j in range( lenght_complementare ) :
                                    func_vals_temp.append( res_temp.func_vals[-1] )
                                res[i].func_vals = func_vals_temp
                                res[i].x_iters = res[i].x_iters + ( [ res[i].x_iters[-1] ] * ( lenght_complementare ) )

                        if( plot == True ):
                            name = plot_name + ".png"
                            self.plot_bayesian_optimization( res, name )
    
        
                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )

            else:
                print("Not implemented \n")

        #Forest Minimize
        if( minimizer == forest_minimize ):
            if( save == False and early_stop == False ):
                for i in range( different_iteration ):
                    res.append( forest_minimize(f, 
                                                bounds,
                                                base_estimator=base_estimator_forest,
                                                n_calls=number_of_call,
                                                acq_func=acq_func,
                                                n_random_starts = n_random_starts,
                                                x0=x0[i],
                                                y0=y0[i],
                                                random_state=random_state ) )
                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )

            elif ( ( save_step >= number_of_call and save == True ) and  ( early_step >= number_of_call or early_stop == False ) ):
                for i in range( different_iteration ):
                    save_name_t = save_name + "_" + str(i) + ".pkl"
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
                                                callback=[checkpoint_saver[i] ] ) )
                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )
            
            elif ( save == True and early_stop == False ):
                for i in range( different_iteration ):
                    save_name_t = save_name + "_" + str(i) + ".pkl"
                    checkpoint_saver[i] = CheckpointSaver( save_name_t ) #save

                    res.append( forest_minimize(f, 
                                            bounds,
                                            base_estimator=base_estimator_forest,
                                            n_calls=save_step,
                                            acq_func=acq_func,
                                            n_random_starts = n_random_starts,
                                            x0=x0[i],
                                            y0=y0[i],
                                            callback=[checkpoint_saver[i] ] ) )

                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )

                number_of_call_r = number_of_call - save_step

                while ( number_of_call_r > 0 ) :
                    if( number_of_call_r >= save_step ):
                        for i in range( different_iteration ):
                            save_name_t = save_name + "_" + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals
                            save_name_t = "./" + save_name + "_" + str(i) + ".pkl"
                            checkpoint_saver_t = CheckpointSaver( save_name_t ) #save

                            res[i] = forest_minimize(f, 
                                                    bounds,
                                                    base_estimator=base_estimator_forest,
                                                    n_calls=save_step,
                                                    acq_func=acq_func,
                                                    n_random_starts = n_random_starts,
                                                    x0=x0_restored, 
                                                    y0=y0_restored,
                                                    callback=[checkpoint_saver[i] ] )

                

                            checkpoint_saver[i] = checkpoint_saver_t
                        if( plot == True ):
                            name = plot_name + ".png"
                            self.plot_bayesian_optimization( res, name )

                        number_of_call_r = number_of_call_r - save_step

                    else:
                        for i in range( different_iteration ):
                            save_name_t = save_name + "_" + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals

                            res[i] = forest_minimize(f, 
                                                    bounds,
                                                    base_estimator=base_estimator_forest,
                                                    n_calls=number_of_call_r,
                                                    acq_func=acq_func,
                                                    n_random_starts = n_random_starts,
                                                    x0=x0_restored, 
                                                    y0=y0_restored,
                                                    callback=[checkpoint_saver[i] ] )

                        if( plot == True ):
                            name = plot_name + ".png"
                            self.plot_bayesian_optimization( res, name )
                        number_of_call_r = number_of_call_r - save_step

            elif ( save == False and early_stop == True ):

                for i in range( different_iteration ):

                    res_temp = forest_minimize(f, 
                                                bounds,
                                                base_estimator=base_estimator_forest,
                                                n_calls=number_of_call,
                                                acq_func=acq_func,
                                                n_random_starts = n_random_starts,
                                                x0=x0[i], 
                                                y0=y0[i],
                                                callback=[ self.MyCustomEarlyStopper( n_stop = early_step ) ] )

                    if( len( res_temp.func_vals ) < number_of_call + len( y0[i] ) ):
                        #print( "EARLY_f" )
                        lenght_complementare = number_of_call - len( res_temp.func_vals ) + len( y0[i] )
                        func_vals_temp = []
                        for i in range( len(res_temp.func_vals) ) :
                            func_vals_temp.append( res_temp.func_vals[i])
                        for i in range( lenght_complementare ) :
                            func_vals_temp.append( res_temp.func_vals[-1] )
                        res_temp.func_vals = func_vals_temp
                        res_temp.x_iters = res_temp.x_iters + ( [ res_temp.x_iters[-1] ] * ( lenght_complementare ) )
                        #print( len(res_temp.func_vals), " res_temp.func_vals ", res_temp.func_vals )
                        #print( len(res_temp.x_iters),  " res_temp.x_iters ", res_temp.x_iters )
                    res.append( res_temp )

                

                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )
    
            elif ( save == True and early_stop == True ):

                for i in range( different_iteration ):
                    save_name_t = save_name + "_" + str(i) + ".pkl"
                    checkpoint_saver[i] = CheckpointSaver( save_name_t ) #save

                    res_temp = forest_minimize(f, 
                                                bounds,
                                                base_estimator=base_estimator_forest,
                                                n_calls=save_step,
                                                acq_func=acq_func,
                                                n_random_starts = n_random_starts,
                                                x0=x0[i], 
                                                y0=y0[i], 
                                                callback=[checkpoint_saver[i], self.MyCustomEarlyStopper( n_stop = early_step ) ] )

                    if( len( res_temp.func_vals ) < save_step + len( y0[i] ) ):
                        lenght_complementare = save_step + len( y0[i] ) - len( res_temp.func_vals )
                        func_vals_temp = []
                        for j in range( len(res_temp.func_vals) ) :
                            func_vals_temp.append( res_temp.func_vals[j])
                        for j in range( lenght_complementare ) :
                            func_vals_temp.append( res_temp.func_vals[-1] )
                        res_temp.func_vals = func_vals_temp
                        res_temp.x_iters = res_temp.x_iters + ( [ res_temp.x_iters[-1] ] * ( lenght_complementare ) )
                    
                    res.append( res_temp )

                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )

                number_of_call_r = number_of_call - save_step

                while ( number_of_call_r > 0 ) :
                    
                    if( number_of_call_r >= save_step ):
                        for i in range( different_iteration ):
                            save_name_t = save_name + "_" + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals
                            save_name_t = "./" + save_name + "_" + str(i) + ".pkl"
                            checkpoint_saver_t = CheckpointSaver( save_name_t ) #save

                            res[i] = dummy_minimize(f, 
                                                bounds, 
                                                n_calls=save_step, 
                                                x0=x0_restored, 
                                                y0=y0_restored,
                                                callback=[checkpoint_saver[i], self.MyCustomEarlyStopper( n_stop = early_step ) ], 
                                                random_state=random_state)

                            checkpoint_saver[i] = checkpoint_saver_t

                        number_of_call_r = number_of_call_r - save_step

                        for i in range( different_iteration ):
                            if( len( res[i].func_vals ) < number_of_call + len( y0[i] ) - number_of_call_r ):
                                #print("before ", res[i].func_vals )
                                lenght_complementare = number_of_call + len( y0[i] ) - number_of_call_r - len( res[i].func_vals )
                                #print("Len_c ", lenght_complementare)
                                #print("Len fun", len( res[i].func_vals ))
                                func_vals_temp = []
                                for j in range( len( res[i].func_vals ) ) :
                                    func_vals_temp.append( res[i].func_vals[j])
                                for j in range( lenght_complementare ) :
                                    func_vals_temp.append( res[i].func_vals[-1] )
                                res[i].func_vals = func_vals_temp
                                res[i].x_iters = res[i].x_iters + ( [ res[i].x_iters[-1] ] * ( lenght_complementare ) )
                                #print("after ", res[i].func_vals )


                        if( plot == True ):
                            name = plot_name + ".png"
                            self.plot_bayesian_optimization( res, name )
                        

                    else:
                        for i in range( different_iteration ):
                            save_name_t = save_name + "_" + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals

                            res[i] = dummy_minimize(f, 
                                                bounds, 
                                                n_calls=number_of_call_r, 
                                                x0=x0_restored, 
                                                y0=y0_restored,
                                                callback=[checkpoint_saver[i], self.MyCustomEarlyStopper( n_stop = early_step ) ], 
                                                random_state=random_state)

                        number_of_call_r = number_of_call_r - save_step

                        for i in range( different_iteration ):
                            if( len( res[i].func_vals ) < number_of_call + len( y0[i] ) ):
                                lenght_complementare = number_of_call - len( res[i].func_vals ) + len( y0[i] )
                                func_vals_temp = []
                                for j in range( len(res_temp.func_vals) ) :
                                    func_vals_temp.append( res_temp.func_vals[j])
                                for j in range( lenght_complementare ) :
                                    func_vals_temp.append( res_temp.func_vals[-1] )
                                res[i].func_vals = func_vals_temp
                                res[i].x_iters = res[i].x_iters + ( [ res[i].x_iters[-1] ] * ( lenght_complementare ) )

                        if( plot == True ):
                            name = plot_name + ".png"
                            self.plot_bayesian_optimization( res, name )
    
        
                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )

            else:
                print("Not implemented \n")
                
        #GP Minimize
        if( minimizer == gp_minimize ):
            if( save == False and early_stop == False ):
                for i in range( different_iteration ):
                    gpr = GaussianProcessRegressor(kernel=kernel, 
                                                alpha=alpha,
                                                normalize_y=True, 
                                                noise="gaussian",
                                                n_restarts_optimizer=2)

                    opt = skopt_optimizer(bounds, 
                                    base_estimator=gpr, 
                                    acq_func=acq_func,
                                    n_random_starts = n_random_starts,
                                    acq_optimizer="sampling", 
                                    random_state=random_state)

                    if( x0[i] != None and y0[i] != None):
                        opt.tell(x0[i], y0[i], fit=True)
                    res.append( opt.run(f, number_of_call) )

                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )

            elif ( ( save_step >= number_of_call and save == True ) and  ( early_step >= number_of_call or early_stop == False )  ):
                for i in range( different_iteration ):
                    
                    gpr = GaussianProcessRegressor(kernel=kernel, 
                                                alpha=alpha ** 2,
                                                normalize_y=True, 
                                                noise="gaussian",
                                                n_restarts_optimizer=2)

                    opt = skopt_optimizer(bounds, 
                                    base_estimator=gpr, 
                                    acq_func=acq_func,
                                    n_random_starts = n_random_starts,
                                    acq_optimizer="sampling", 
                                    random_state=random_state)

                    if( x0[i] != None and y0[i] != None):
                        opt.tell(x0[i], y0[i], fit=True)

                    res_t = opt.run(f, number_of_call)
                    res.append( res_t )

                checkpoint_saver = self.dump_BO( res, save_name ) #save

                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )

            elif ( save == True and early_stop == False ):
                
                for i in range( different_iteration ):
                    gpr = GaussianProcessRegressor(kernel=kernel, 
                                                alpha=alpha ** 2,
                                                normalize_y=True, 
                                                noise="gaussian",
                                                n_restarts_optimizer=2)

                    opt = skopt_optimizer(bounds, 
                                    base_estimator=gpr, 
                                    acq_func=acq_func,
                                    n_random_starts = n_random_starts,
                                    acq_optimizer="sampling", 
                                    random_state=random_state)

                    if( x0[i] != None and y0[i] != None):
                        opt.tell(x0[i], y0[i], fit=True)

                    res_t = opt.run(f, save_step)
                    res.append( res_t )

                checkpoint_saver = self.dump_BO( res, save_name ) #save
                number_of_call_r = number_of_call - save_step

                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )
                

                while ( number_of_call_r > 0 ) :
                    if( number_of_call_r >= save_step ):
                        partial_res = self.load_BO( checkpoint_saver ) #restore

                        for i in range( different_iteration ):
                            x0_restored = partial_res[i].x_iters
                            y0_restored = list(partial_res[i].func_vals)

                            gpr = GaussianProcessRegressor(kernel=kernel, 
                                                alpha=alpha ** 2,
                                                normalize_y=True, 
                                                noise="gaussian",
                                                n_restarts_optimizer=2)

                            opt = skopt_optimizer(bounds, 
                                            base_estimator=gpr, 
                                            acq_func=acq_func,
                                            n_random_starts = 0,
                                            acq_optimizer="sampling", 
                                            random_state=random_state)

                            opt.tell(x0_restored, y0_restored, fit=True)

                            res_t = opt.run(f, save_step)
                            res[i] = res_t

                        checkpoint_saver = self.dump_BO( res, save_name ) #save
                        number_of_call_r = number_of_call_r - save_step

                        if( plot == True ):
                            name = plot_name + ".png"
                            self.plot_bayesian_optimization( res, name )

                    else:
                        partial_res = self.load_BO( checkpoint_saver ) #restore
                        for i in range( different_iteration ):
                            x0_restored = partial_res[i].x_iters
                            y0_restored = list(partial_res[i].func_vals)

                            gpr = GaussianProcessRegressor(kernel=kernel, 
                                                alpha=alpha ** 2,
                                                normalize_y=True, 
                                                noise="gaussian",
                                                n_restarts_optimizer=2)

                            opt = skopt_optimizer(bounds, 
                                            base_estimator=gpr, 
                                            acq_func=acq_func,
                                            n_random_starts = 0,
                                            acq_optimizer="sampling", 
                                            random_state=random_state)

                            opt.tell(x0_restored, y0_restored, fit=True)

                            res_t = opt.run(f, number_of_call_r)
                            res[i] = res_t

                        checkpoint_saver = self.dump_BO( res, save_name ) #save
                        number_of_call_r = number_of_call_r - save_step

                        if( plot == True ):
                            name = plot_name + ".png"
                            self.plot_bayesian_optimization( res, name )

            #TO DO    callback=[ MyCustomEarlyStopper( n_stop = early_step ) ]
            elif ( save == False and early_stop == True ):

                for i in range( different_iteration ):
                    gpr = GaussianProcessRegressor(kernel=kernel, 
                                                alpha=alpha ** 2,
                                                normalize_y=True, 
                                                noise="gaussian",
                                                n_restarts_optimizer=2)

                    opt = skopt_optimizer(bounds, 
                                    base_estimator=gpr, 
                                    acq_func=acq_func,
                                    n_random_starts = n_random_starts,
                                    acq_optimizer="sampling", 
                                    random_state=random_state
                                    #callback=[ self.MyCustomEarlyStopper( n_stop = early_step ) ]
					)

                    if( x0[i] != None and y0[i] != None):
                        opt.tell(x0[i], y0[i], fit=True)
                    res.append( opt.run(f, number_of_call) )
                

                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )
    
            #TO DO
            elif ( save == True and early_stop == True ):

                for i in range( different_iteration ):
                    save_name_t = save_name + "_" + str(i) + ".pkl"
                    checkpoint_saver[i] = CheckpointSaver( save_name_t ) #save

                    res_temp = forest_minimize(f, 
                                                bounds,
                                                base_estimator=base_estimator_forest,
                                                n_calls=save_step,
                                                acq_func=acq_func,
                                                n_random_starts = n_random_starts,
                                                x0=x0[i], 
                                                y0=y0[i], 
                                                callback=[checkpoint_saver[i], self.MyCustomEarlyStopper( n_stop = early_step ) ] )

                    if( len( res_temp.func_vals ) < save_step + len( y0[i] ) ):
                        lenght_complementare = save_step + len( y0[i] ) - len( res_temp.func_vals )
                        func_vals_temp = []
                        for j in range( len(res_temp.func_vals) ) :
                            func_vals_temp.append( res_temp.func_vals[j])
                        for j in range( lenght_complementare ) :
                            func_vals_temp.append( res_temp.func_vals[-1] )
                        res_temp.func_vals = func_vals_temp
                        res_temp.x_iters = res_temp.x_iters + ( [ res_temp.x_iters[-1] ] * ( lenght_complementare ) )
                    
                    res.append( res_temp )

                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )

                number_of_call_r = number_of_call - save_step

                while ( number_of_call_r > 0 ) :
                    
                    if( number_of_call_r >= save_step ):
                        for i in range( different_iteration ):
                            save_name_t = save_name + "_" + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals
                            save_name_t = "./" + save_name + "_" + str(i) + ".pkl"
                            checkpoint_saver_t = CheckpointSaver( save_name_t ) #save

                            res[i] = dummy_minimize(f, 
                                                bounds, 
                                                n_calls=save_step, 
                                                x0=x0_restored, 
                                                y0=y0_restored,
                                                callback=[checkpoint_saver[i], self.MyCustomEarlyStopper( n_stop = early_step ) ], 
                                                random_state=random_state)

                            checkpoint_saver[i] = checkpoint_saver_t

                        number_of_call_r = number_of_call_r - save_step

                        for i in range( different_iteration ):
                            if( len( res[i].func_vals ) < number_of_call + len( y0[i] ) - number_of_call_r ):
                                #print("before ", res[i].func_vals )
                                lenght_complementare = number_of_call + len( y0[i] ) - number_of_call_r - len( res[i].func_vals )
                                #print("Len_c ", lenght_complementare)
                                #print("Len fun", len( res[i].func_vals ))
                                func_vals_temp = []
                                for j in range( len( res[i].func_vals ) ) :
                                    func_vals_temp.append( res[i].func_vals[j])
                                for j in range( lenght_complementare ) :
                                    func_vals_temp.append( res[i].func_vals[-1] )
                                res[i].func_vals = func_vals_temp
                                res[i].x_iters = res[i].x_iters + ( [ res[i].x_iters[-1] ] * ( lenght_complementare ) )
                                #print("after ", res[i].func_vals )


                        if( plot == True ):
                            name = plot_name + ".png"
                            self.plot_bayesian_optimization( res, name )
                        

                    else:
                        for i in range( different_iteration ):
                            save_name_t = save_name + "_" + str(i) + ".pkl"
                            partial_res = load( save_name_t )  #restore
                            x0_restored = partial_res.x_iters
                            y0_restored = partial_res.func_vals

                            res[i] = dummy_minimize(f, 
                                                bounds, 
                                                n_calls=number_of_call_r, 
                                                x0=x0_restored, 
                                                y0=y0_restored,
                                                callback=[checkpoint_saver[i], self.MyCustomEarlyStopper( n_stop = early_step ) ], 
                                                random_state=random_state)

                        number_of_call_r = number_of_call_r - save_step

                        for i in range( different_iteration ):
                            if( len( res[i].func_vals ) < number_of_call + len( y0[i] ) ):
                                lenght_complementare = number_of_call - len( res[i].func_vals ) + len( y0[i] )
                                func_vals_temp = []
                                for j in range( len(res_temp.func_vals) ) :
                                    func_vals_temp.append( res_temp.func_vals[j])
                                for j in range( lenght_complementare ) :
                                    func_vals_temp.append( res_temp.func_vals[-1] )
                                res[i].func_vals = func_vals_temp
                                res[i].x_iters = res[i].x_iters + ( [ res[i].x_iters[-1] ] * ( lenght_complementare ) )

                        if( plot == True ):
                            name = plot_name + ".png"
                            self.plot_bayesian_optimization( res, name )
    
        
                if( plot == True ):
                    name = plot_name + ".png"
                    self.plot_bayesian_optimization( res, name )

            else:
                print("Not implemented \n")

        return res

    def median(self, list_of_res):
        """
            Given a Bayesian_optimization result 
            the median of the min y found
            -statistics module needed

            Parameters
            ----------
            list_of_res : A Bayesian_optimization result

            Returns
            -------
            val : The median of the min y found
        """    
        r = []
        for res in list_of_res:
            r.append( list(self.convergence_res(res)) )
        val = []
        for i in r:
            val.append( i[-1] )
        val = statistics.median( val )
        return val

    def convergence_res(self, res):
        """
            Given a single element of a
            Bayesian_optimization return the 
            convergence of y

            Parameters
            ----------
            res : A single element of a 
                Bayesian_optimization result

            Returns
            -------
            val : A list with the best min seen for 
                each evaluation
        """    
        val = res.func_vals
        for i in range( len(val) ):
            if( i != 0 and val[i] > val[i-1] ):
                val[i] = val[i-1]
        return val

    def total_mean(self, list_of_res):
        """
            Given a Bayesian_optimization result 
            return a list of the mean with the other 
            tests runned 

            Parameters
            ----------
            list_of_res : A Bayesian_optimization result

            Returns
            -------
            media : A list of the mean with the other 
                    tests runned
        """    
        r = []
        for res in list_of_res:
            r.append( list(self.convergence_res(res)) )
        a = []
        media = []
        for i in range( len( list_of_res[0].func_vals ) ):
            for j in range( default_parameters["different_iteration"] ):
                a.append( r[j][i] )
            media.append( np.mean(a, dtype=np.float64) )
            a = []
        return media

    def total_standard_deviation(self, list_of_res):
        """
            Given a Bayesian_optimization result 
            return a list of the standard deviation
            with the other tests runned 

            Parameters
            ----------
            list_of_res : A Bayesian_optimization result

            Returns
            -------
            dev : A list of the standard deviation
                with the other test runned
        """    
        r = []
        for res in list_of_res:
            r.append( list(self.convergence_res(res)) )
        a = []
        dev = []
        for i in range( len( list_of_res[0].func_vals ) ):
            for j in range( default_parameters["different_iteration"] ): #prima era n_test
                a.append( r[j][i] )
            dev.append( np.std(a, dtype=np.float64) )
            a = []
        return dev

    def upper_standard_deviation(self, list_of_res):
        """
            Given a Bayesian_optimization result 
            return a list of the higher standard 
            deviation from the tests runned 

            Parameters
            ----------
            list_of_res : A Bayesian_optimization result

            Returns
            -------
            upper : A list of the higher standard 
                    deviation with the other tests runned
        """
        media = self.total_mean(list_of_res)
        dev = self.total_standard_deviation(list_of_res)
        upper = []
        for i in range( len( media ) ):
            upper.append( media[i] + dev[i] ) 
        return upper

    def lower_standard_deviation(self, list_of_res):
        """
            Given a Bayesian_optimization result 
            return a list of the lower standard 
            deviation from the tests runned 

            Parameters
            ----------
            list_of_res : A Bayesian_optimization result

            Returns
            -------
            lower : A list of the lower standard 
                    deviation with the other tests runned
        """
        media = self.total_mean(list_of_res)
        dev = self.total_standard_deviation(list_of_res)
        lower = []
        for i in range( len( media ) ):
            lower.append( media[i] - dev[i] ) 
        return lower

    def convergence_res_x(self, res, r_min):
        """
            Given a single element of a
            Bayesian_optimization and the argmin
            of the function return the convergence of x
            centred around the lowest distance 
            from the argmin
            -scipy.spatial.distance module needed


            Parameters
            ----------
            res : A single element of a 
                Bayesian_optimization result

            min : the argmin of the function in form 
                of a list as it follows:
                -[[-pi, 12.275], [pi, 2.275], [9.42478, 2.475] ]

            Returns
            -------
            distance : A list with the distance between 
                the best x seen for each evaluation
                and the argmin
        """    
        val = res.x_iters
        distance = []
        if( len(r_min) == 1 ):
            for i in range( len(val) ):
                if( i != 0 and dist_eu.euclidean(val[i],r_min) > distance[i-1] ):
                    distance.append( distance[i-1] )
                else:
                    distance.append( dist_eu.euclidean(val[i],r_min) )
            return distance
        else:
            distance_all_min = []
            for i in range( len(val) ):
                for j in range( len(r_min) ):
                    distance_all_min.append( dist_eu.euclidean(val[i],r_min[j]) )
                min_distance = min( distance_all_min )
                if( i != 0 and min_distance > distance[i-1] ):
                    distance.append( distance[i-1] )
                else:
                    distance.append( min_distance )
                distance_all_min = []
            return distance

    def total_mean_x(self, list_of_res, min):
        """
            Given a Bayesian_optimization result
            and the argmin of the function return 
            the mean of x centred around the lowest 
            distance from the argmin

            Parameters
            ----------
            res : A Bayesian_optimization result

            min : the argmin of the function in form 
                of a list as it follows:
                -[[-pi, 12.275], [pi, 2.275], [9.42478, 2.475] ]

            Returns
            -------
            media : A list with the mean of the distance 
                    between the best x seen for each 
                    evaluation and the argmin
        """
        r = []
        for res in list_of_res:
            r.append( list(self.convergence_res_x(res, min)) )
        a = []
        media = []
        for i in range( len( list_of_res[0].func_vals ) ):
            for j in range( default_parameters["different_iteration"] ):
                a.append( r[j][i] )
            media.append( np.mean(a, dtype=np.float64) )
            a = []
        return media

    def total_standard_deviation_x(self, list_of_res, min):
        """
            Given a Bayesian_optimization result
            and the argmin of the function return 
            the standard deviation of x centred around 
            the lowest distance from the argmin

            Parameters
            ----------
            res : A Bayesian_optimization result

            min : the argmin of the function in form 
                of a list as it follows:
                -[[-pi, 12.275], [pi, 2.275], [9.42478, 2.475] ]

            Returns
            -------
            dev : A list with the standard deviation of the 
                distance between the best x seen for each 
                evaluation and the argmin
        """
        r = []
        for res in list_of_res:
            r.append( list(self.convergence_res_x(res, min)) )
        a = []
        dev = []
        for i in range( len( list_of_res[0].func_vals ) ):
            for j in range( default_parameters["different_iteration"] ):
                a.append( r[j][i] )
            dev.append( np.std(a, dtype=np.float64) )
            a = []
        return dev

    def upper_standard_deviation_x(self, list_of_res, min):
        """
            Given a Bayesian_optimization result
            and the argmin of the function return 
            higher standard deviation from the tests 
            runned in x

            Parameters
            ----------
            res : A Bayesian_optimization result

            min : the argmin of the function in form 
                of a list as it follows:
                -[[-pi, 12.275], [pi, 2.275], [9.42478, 2.475] ]

            Returns
            -------
            upper : A list with the higher standard deviation 
                    from the tests runned in x
        """
        media = self.total_mean_x(list_of_res, min)
        dev = self.total_standard_deviation_x(list_of_res, min)
        upper = []
        for i in range( len( media ) ):
            upper.append( media[i] + dev[i] ) 
        return upper

    def lower_standard_deviation_x(self, list_of_res, min):
        """
            Given a Bayesian_optimization result
            and the argmin of the function return 
            lower standard deviation from the tests 
            runned in x

            Parameters
            ----------
            res : A Bayesian_optimization result

            min : the argmin of the function in form 
                of a list as it follows:
                -[[-pi, 12.275], [pi, 2.275], [9.42478, 2.475] ]

            Returns
            -------
            lower : A list with the lower standard deviation 
                    from the tests runned in x
        """
        media = self.total_mean_x(list_of_res, min)
        dev = self.total_standard_deviation_x(list_of_res, min)
        lower = []
        for i in range( len( media ) ):
            lower.append( media[i] - dev[i] ) 
        return lower

    def my_key_fun(self, res ):
        """
            Sort key for fun_min function
        """    
        return res.fun

    def fun_min(self, list_of_res ):
        """
            Return the min of a list of BO
        """    
        min_res = min(list_of_res, key = self.my_key_fun )
        return [ min_res.fun, min_res.x ]

    def tabella(self, list_of_list_of_res ):
        """
            Given a list of Bayesian_optimization results
            return a list with name, mean, median, 
            standard deviation and min result founded
            for each Bayesian_optimization result

            Parameters
            ----------
            list_of_list_of_res : A list of Bayesian_optimization 
                                results 

            Returns
            -------
            lista : A list with name, mean, median, 
                    standard deviation and min result founded
                    for each Bayesian_optimization result
        """    
        lista = []
        for i in list_of_list_of_res:
            fun_media = []
            for it in range( default_parameters["different_iteration"] ):
                fun_media.append( i[0][it].fun )
            
            lista.append( [ i[1], np.mean(fun_media, dtype=np.float64) , self.median( i[0] ), np.std(fun_media, dtype=np.float64), self.fun_min( i[0] ) ] )
            # nome, media, mediana, std, [.fun min, .x min]
        return lista

    def my_key_sort(self, list_with_name):
        """
            Sort key for top_5 funcion
        """    
        return list_with_name[0]

    def top_5(self, list_of_list_of_res ):
        """
            Given a list of Bayesian_optimization results
            find out the best 5 result confronting the 
            best mean result

            Parameters
            ----------
            list_of_list_of_res : A list of Bayesian_optimization 
                                results 

            Returns
            -------
            list_medie : A list of each .pkl file's name 
                        just saved
                        -    list_of_list_of_res = [[res_BO_1,"name_1", 1], [res_BO_2,"name_2", 2],etc.]
        """    
        list_medie = []
        for i in list_of_list_of_res:
            list_medie.append( [ self.total_mean( i[0] ), i[1], i[2] ] )
        list_medie.sort( key = self.my_key_sort )
        list_medie = list_medie[:5]
        return list_medie

    def dump_BO(self, list_of_res, stringa = 'result' ):
        """
            Dump (save) the Bayesian_optimization result

            Parameters
            ----------
            list_of_res : A result of a Bayesian_optimization
                        run
            stringa : Name of the log file saved in .pkl 
                    format after the run of the function

            Returns
            -------
            lista_dump : A list of each .pkl file's name 
                        just saved 
        """
        lista_dump = []
        for n in range( len( list_of_res ) ):
            name_file = stringa + str( n ) + '.pkl'
            dump( list_of_res[n] , name_file)
            lista_dump.append( name_file )
        return lista_dump

    def load_BO(self, lista_dump ):
        """
            Load a list of pkl files, it should have the 
            list returned from dump_BO to work 
            properly, as it follows:
            -   lista_dump = dump_BO( res_gp_rosenbrock )
            -   res_loaded = load_BO( lista_dump )

            Parameters
            ----------
            lista_dump : A list of .pkl files

            Returns
            -------
            lista_res_loaded : A Bayesian_optimization result
        """
        lista_res_loaded = []
        for n in range( len( lista_dump ) ):
            res_loaded = load( lista_dump[n] )
            lista_res_loaded.append( res_loaded )
        return lista_res_loaded

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

        print("Start Bayesian Optimization")

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
                            X0 = default_parameters["x0"],
                            Y0 = default_parameters["y0"],
                            n_random_starts = default_parameters["n_random_starts"],
                            save = default_parameters["save"],
                            save_step = default_parameters["save_step"],
                            save_name = default_parameters["save_name"],
                            early_stop = default_parameters["early_stop"],
                            early_step = default_parameters["early_step"],
                            plot = default_parameters["plot"],
                            plot_name = default_parameters["plot_name"]
        )

        # To have the right result
        if self.optimization_type == 'Maximize':
            for i in range( len(optimize_result) ):
                optimize_result[i].fun = - optimize_result[i].fun
                for j in range( len(optimize_result[i].func_vals) ):
                    optimize_result[i].func_vals[j] = - optimize_result[i].func_vals[j]



        # Create Best_evaluation object from optimization results
        result = Best_evaluation(self.hyperparameters,
                                 optimize_result,
                                 self._iterations,
                                 self.metric.__class__.__name__)


        return result