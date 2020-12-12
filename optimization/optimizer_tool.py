import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skopt.learning import GaussianProcessRegressor, RandomForestRegressor, ExtraTreesRegressor
from skopt import Optimizer as skopt_optimizer
from skopt.utils import dimensions_aslist

def choose_optimizer(params,restart=False):

    params_space_list=dimensions_aslist(params.search_space)

    #### Choice of the surrogate model
    # Random forest
    if params.surrogate_model == "RF":
        estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=3,random_state=params.random_state)
    # Extra Tree
    elif params.surrogate_model == "ET":
        estimator = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=3,random_state=params.random_state)
        # GP Minimize
    elif params.surrogate_model == "GP":
        estimator = GaussianProcessRegressor(kernel=params.kernel, random_state=params.random_state)
        # Random Search
    elif params.surrogate_model == "RS":
        estimator = "dummy"

    if restart==False:
        opt = skopt_optimizer(params_space_list, base_estimator=estimator,
                              acq_func=params.acq_func,
                              acq_optimizer='sampling',
                              n_initial_points=params.n_random_starts,
                              initial_point_generator=params.initial_point_generator,
                              # work only for version skopt 8.0!!!
                              acq_optimizer_kwargs={"n_points": 10000, "n_restarts_optimizer": 5, "n_jobs": 1},
                              acq_func_kwargs={"xi": 0.01, "kappa": 1.96},
                              random_state=params.random_state)
    else:
        opt = skopt_optimizer(params_space_list, base_estimator=estimator,
                              acq_func=params.acq_func,
                              acq_optimizer='sampling',
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

def plot_model_runs(model_runs,current_call, name_plot):
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

    values=[model_runs["iteration_"+str(i)] for i in range(current_call+1)]

    plt.ioff()
    plt.xlabel('number of calls')
    plt.grid(True)
    plt.boxplot(values)

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

##############################################################################
class BestEvaluation:
    
    def __init__(self,params,resultsBO,times):
        """
        Create an object with all the information about Bayesian Optimization
        
        """
        search_space=params.search_space
        optimization_type=params.optimization_type
        n_calls=len(resultsBO.func_vals)
                
        #Info about optimization
        self.info=dict()
        self.info.update({"dataset_name":params.dataset.get_metadata()["info"]["name"]})
        self.info.update({"dataset_path":params.dataset.path})  
        self.info.update({"metric_name":params.metric.__class__.__name__})      
        self.info.update({"kernel":str(params.kernel)})
        self.info.update({"number of calls":n_calls}) 
        self.info.update({"acq_func":params.acq_func})
        self.info.update({"surrogate_model":params.surrogate_model})
        self.info.update({"optimization_type":"Maximize" if optimization_type=="Maximize" else "Minimize"})
        self.info.update({"model_runs":params.model_runs})
        self.info.update({"save_models":params.save_models})
        self.info.update({"save_step":params.save_step})
        self.info.update({"save_name":params.save_name})
        self.info.update({"save_path":params.save_path})
        self.info.update({"early_stop":params.early_stop})
        self.info.update({"early_step":params.early_step})        
        self.info.update({"plot_model":params.plot_model})
        self.info.update({"plot_best_seen":params.plot_best_seen})
        self.info.update({"plot_name":params.plot_name})
        self.info.update({"log_scale_plot":params.log_scale_plot})  
        self.info.update({"search_space":str(params.search_space)})
        self.info.update({"metric_name":params.metric.__class__.__name__})
        self.info.update({"metric_attributes":params.metric.parameters})

        #Reverse the sign of minimization if the problem is a maximization    
        if optimization_type=="Maximize":
            self.func_vals=[-val for val in resultsBO.func_vals]
            self.y_best=resultsBO.fun
        else:
            self.func_vals=[val for val in resultsBO.func_vals]
            self.y_best=resultsBO.fun

        self.x_iters=dict()
        name_hyperparameters=list(search_space.keys())
        
        #dictionary of x_iters
        lenList=len(resultsBO.x_iters)
        for i,name in enumerate(name_hyperparameters):
            self.x_iters.update({name: [resultsBO.x_iters[j][i] for j in range(lenList)]})   

        self.times=times
        self.dict_model_runs= params.dict_model_runs
        self.metric=params.metric
        self.extra_metrics=params.extra_metrics    
        self.search_space=params.search_space
        
    def save(self,name_file):
        """
        Save results for Bayesian Optimization
        """
        
        Results=dict()
        Results.update(self.info)
                
        Results.update({"time":self.times})
        Results.update({"dict_model_runs":self.dict_model_runs})
        Results.update({"x_iters":self.x_iters})
        Results.update({"f_val":self.func_vals})
        
        with open(name_file, 'w') as fp:
            json.dump(Results, fp)

    def save_to_csv(self,name_file):     

        n_row = len(self.func_vals)
        n_extra_metrics=len(self.extra_metrics)
        
        #creation of the Dataframe 
        df = pd.DataFrame()       
        df['dataset'] = [self.info["dataset_name"]] * n_row
        df['surrogate model'] = [self.info["surrogate_model"]] * n_row
        df['acquisition function'] = [self.info["acq_func"]] * n_row
        df['num_iteration']=[i for i in range(n_row)] 
        df['time'] = [self.times[i] for i in range(n_row)]  
        df['Mean(model_runs)'] = [np.mean(self.dict_model_runs[self.metric.__class__.__name__]['iteration_'+str(i)]) for i in range(n_row)]    
        df['Standard_Deviation(model_runs)'] = [np.std(self.dict_model_runs[self.metric.__class__.__name__]['iteration_'+str(i)]) for i in range(n_row)]    

        for hyperparameter in list(self.x_iters.keys()):
            df[hyperparameter] = self.x_iters[hyperparameter]  
            
        for metric,i in zip(self.extra_metrics,range(n_extra_metrics)):
            try:
                df[metric.info()["name"]+'(not optimized)']=[np.median(self.dict_model_runs[metric.__class__.__name__]['iteration_'+str(i)]) for i in range(n_row)]    
            except:
                df[metric.__class__.__name__+'(not optimized)']=[np.median(self.dict_model_runs[metric.__class__.__name__]['iteration_'+str(i)]) for i in range(n_row)]    

        if not name_file.endswith(".csv"):
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