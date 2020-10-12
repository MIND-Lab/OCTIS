import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from skopt.learning import GaussianProcessRegressor, RandomForestRegressor, ExtraTreesRegressor
from skopt import Optimizer as skopt_optimizer
from skopt.utils import dimensions_aslist

def choose_optimizer(params):

    params_space_list=dimensions_aslist(params.search_space)

    #### Choice of the surrogate model
    # Random forest
    if params.surrogate_model == "RF":
        estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=3,random_state=params.random_state)
        #surrogate_model_name = "random_forest"
    # Extra Tree
    elif params.surrogate_model == "ET":
        estimator = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=3,random_state=params.random_state)
        #surrogate_model_name = "extra tree regressor"
        # GP Minimize
    elif params.surrogate_model == "GP":
        estimator = GaussianProcessRegressor(kernel=params.kernel, random_state=params.random_state)
        #surrogate_model_name = "gaussian process"
        # Random Search
    elif params.surrogate_model == "RS":
        estimator = "dummy"
        #surrogate_model_name = "random_minimize"

    opt = skopt_optimizer(params_space_list, base_estimator=estimator,
                          acq_func=params.acq_func,
                          n_initial_points=params.n_random_starts,
                          initial_point_generator=params.initial_point_generator,
                          # work only for version skopt 8.0!!!
                          # acq_optimizer="sampling",
                          acq_optimizer_kwargs={"n_points": 10000, "n_restarts_optimizer": 5, "n_jobs": 1},
                          acq_func_kwargs={"xi": 0.01, "kappa": 1.96},
                          random_state=params.random_state)


    return opt

def convergence_res(values,optimization_type="minimize"):
    """
        Given a single element of a
        Bayesian_optimization return the 
        convergence of y

        Parameters
        ----------
        values : values obtained by BO

        Returns
        -------
        val : A list with the best min seen for 
            each evaluation
    """
    values2=values.copy()
    
    if optimization_type=="minimize":
        for i in range(1,len(values2)):
            if values2[i] > values2[i - 1]:
                values2[i] = values2[i - 1]
    else:
        for i in range(1,len(values2)):
            if values2[i] < values2[i - 1]:
                values2[i] = values2[i - 1]       
    return values2


def early_condition(values, n_stop, n_random):
    """
        Compute the decision to stop or not.

        Parameters
        ----------
        values : values obtained by BO
        
        n_stop : Range of points without improvement

        n_random : Random starting point

        Returns
        -------
        decision : Return True if early stop condition has been reached
    """
    n_min_len = n_stop + n_random
    
    if len(values) >= n_min_len:
        values = convergence_res(values,optimization_type="minimize")
        worst = values[len(values) - (n_stop)]
        best = values[-1]
        diff = worst - best
        if diff == 0:
            return True

    return False

def plot_model_runs(matrix, name_plot):
    """
        Save a boxplot of the data.
        Works only when optimization_runs is 1.

        Parameters
        ----------
        matrix: list of list of list of numbers
                or a 3D matrix
        
        name_plot : The name of the file you want to 
                    give to the plot

    """

    plt.ioff()
    plt.xlabel('number of calls')
    plt.grid(True)
    plt.boxplot(matrix.transpose())

    plt.savefig( name_plot+".png")  

    plt.close()


def plot_bayesian_optimization(values, name_plot,
                               log_scale=False,  conv_max=True):
    """
        Save a convergence plot of the result of a 
        Bayesian_optimization.

        Parameters
        ----------
        values : values obtained by BO

        name_plot : The name of the file you want to 
                    give to the plot

        log_scale : y log scale if True

        conv_min : If True the convergence is for the min,
                    If False is for the max

    """
    if conv_max:
        #minimization problem -->maximization problem
        values=[-val for val in values] 
        media = convergence_res(values,optimization_type="maximize")
        xlabel='max f(x) after n calls'
    else:
        #minimization problem
        media = convergence_res(values,optimization_type="minimize")
        xlabel='min f(x) after n calls'
        
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
    
    plt.savefig(name_plot+".png")  

    plt.close()

def save_csv(name_csv,res,
             matrix_model_runs,
             extra_metrics,        
             dataset_name, 
             hyperparameters_name, 
             metric_name, 
             surrogate_model_name, 
             acquisition_function_name, 
             times, 
             ):
    """
        Create a csv file to describe the topic model optimization.

        Input parameters
        ----------
        name_csv                : [string] name of the .csv file

        res                     :  results of BO 

        matrix_model_runs       : Tensor of metrics computed through BO (1+number of extra metrics,number of calls,number_of_runs)

        extra_metrics           : List of extra metrics names 

        dataset_name            : [string] name of the dataset

        hyperparameters_name    : [list of string] name of the hyperparameters

        metric_name             : [string] name of the metric optimized

        surrogate_model_name    : [string] surrogate model used

        acquisition_function_name : [string] acquisition function used

        times                   : list of time for each point evaluated by Bayesian_optimization
        
        """

    n_row = len(res.func_vals)
    n_extra_metrics=matrix_model_runs.shape[0]-1
    
    #creation of the Dataframe 
    df = pd.DataFrame()

    df['DATASET'] = [dataset_name] * n_row
    for hyperparameter,j in zip(hyperparameters_name,range(len(hyperparameters_name))):
        df[hyperparameter] = [res.x_iters[i][j] for i in range(n_row)]  
    df['SURROGATE'] = [surrogate_model_name] * n_row
    df['ACQUISITION FUNC'] = [acquisition_function_name] * n_row
    df['NUM_ITERATION']=[i for i in range(n_row)] 
    df[metric_name + '(optimized)']=np.median(matrix_model_runs[0,:,:],axis=1)  
    df['TIME'] = [times[i] for i in range(n_row)]           
    df['Mean(model_runs)'] = np.mean(matrix_model_runs[0,:,:],axis=1)
    df['Standard_Deviation(model_runs)'] = np.std(matrix_model_runs[0,:,:],axis=1)
    for metric,i in zip(extra_metrics,range(n_extra_metrics)):
        try:
            df[metric.info()["name"]+'(not optimized)']=np.median(matrix_model_runs[i+1,:,:],axis=1)    
        except:
            df[metric.__class__.__name__+'(not optimized)']=np.median(matrix_model_runs[i+1,:,:],axis=1)   
    #save the Dataframe to a csv
    df.to_csv(name_csv+".csv", index=False, na_rep='Unkown')

##############################################################################
class BestEvaluation:
    
    def __init__(self,params,resultsBO,times):
        """
        Create an object with all the information about Bayesian Optimization
        
        """
        search_space=params.search_space
        matrix_model_runs=params.matrix_model_runs
        extra_metrics=params.extra_metrics
        optimization_type=params.optimization_type

        n_calls=matrix_model_runs.shape[1]
        n_runs=matrix_model_runs.shape[2]        
 
        #Info about optimization
        self.info=dict()
        self.info.update({"dataset name":params.dataset.get_metadata()["info"]["name"]})
        self.info.update({"metric name":params.metric.__class__.__name__})
        self.info.update({"surrogate model":params.surrogate_model})
        self.info.update({"kernel":params.kernel})
        self.info.update({"acquisition function":params.acq_func})
        self.info.update({"number of calls":n_calls})
        self.info.update({"number of model runs":n_runs})
        self.info.update({"type_of optimization":"Maximize" if optimization_type=="Maximize" else "Minimize"})

        #Reverse the sign of minimization if the problem is a maximization    
        self.x_iters=resultsBO.x_iters
        if optimization_type=="Maximize":
            self.func_vals=[-val for val in resultsBO.func_vals]     
            self.y_best=-resultsBO.fun                                         #Best value
        else:
            self.func_vals=[val for val in resultsBO.func_vals]
            self.y_best=resultsBO.fun                                          #Best value

        self.x_iters_as_dict=dict()
        name_hyperparameters=list(search_space.keys())
        
        #dictionary of x_iters
        i=0
        lenList=len(resultsBO.x_iters)
        for name in name_hyperparameters:
            self.x_iters_as_dict.update({name: [resultsBO.x_iters[j][i] for j in range(lenList)]}) 
            i=i+1    

        self.x_best=resultsBO.x                                                #Best x
        self.models_runs= dict(("iteration_"+str(i), list(matrix_model_runs[0,i,:])) for i in range(n_calls))

        #extra metrics info
        self.extra_metrics=dict()
        j=1
        for metric in extra_metrics:
            d={metric.__class__.__name__:dict(("iteration_"+str(i), list(matrix_model_runs[j,i,:])) for i in range(n_calls))}
            self.extra_metrics.update(d)
            j=j+1

        self.times= times 
            
    def save(self,name_file):
        """
        Save results for Bayesian Optimization
        """
        
        Results=dict()
        Results.update({"dataset name":self.info["dataset name"]})
        Results.update({"metric name":self.info["metric name"]})
        Results.update({"surrogate model":self.info["surrogate model"]})
        Results.update({"acquisition function":self.info["acquisition function"]})
        Results.update({"number of calls":self.info["number of calls"]})
        Results.update({"number of model runs":self.info["number of model runs"]})
        Results.update({"type_of optimization":self.info["type_of optimization"]})
        Results.update({"function evaluations":self.func_vals})
        Results.update({"best function value":self.y_best})
        Results.update({"xvals":self.x_iters_as_dict})
        Results.update({"best point":self.x_best})
        Results.update({"time":self.times})
        Results.update({"model runs":self.models_runs})
        Results.update({"extra_metrics":self.extra_metrics})
        
        with open(name_file, 'w') as fp:
            json.dump(Results, fp)


    def save_to_csv(self,name_file):
 
        n_row = len(self.func_vals)
        n_extra_metrics=len(self.extra_metrics)
        
        #creation of the Dataframe 
        df = pd.DataFrame()       
        df['DATASET'] = [self.info["dataset name"]] * n_row
        df['SURROGATE'] = [self.info["surrogate model"]] * n_row
        df['ACQUISITION FUNC'] = [self.info["acquisition function"]] * n_row
        df['NUM_ITERATION']=[i for i in range(n_row)] 
        df['TIME'] = [self.times[i] for i in range(n_row)]  
        df['Mean(model_runs)'] = [np.mean(self.models_runs['iteration_'+str(i)]) for i in range(n_row)]    
        df['Standard_Deviation(model_runs)'] = [np.std(self.models_runs['iteration_'+str(i)]) for i in range(n_row)]    

        for hyperparameter in list(self.x_iters_as_dict.keys()):
            df[hyperparameter] = self.x_iters_as_dict[hyperparameter]  
            
        for metric,i in zip(self.extra_metrics_names,range(n_extra_metrics)):
            try:
                df[metric.info()["name"]+'(not optimized)']=[np.median(self.extra_metrics['iteration_'+str(i)]) for i in range(n_row)]    
            except:
                df[metric.__class__.__name__+'(not optimized)']=[np.median(self.extra_metrics['iteration_'+str(i)]) for i in range(n_row)]    

        if not self.name_file.endswith(".csv"):
            name_file=name_file+".csv"

        #save the Dataframe to a csv        
        df.to_csv(name_file, index=False, na_rep='Unkown')


    def load(self,name):
        """
        Load results for Bayesian Optimization
        """
        with open(name, 'rb') as file:
            result=json.load(file)    
            
        return result