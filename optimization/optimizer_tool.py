import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

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

    if optimization_type=="minimize":
        for i in range(1,len(values)):
            if values[i] > values[i - 1]:
                values[i] = values[i - 1]
    else:
        for i in range(1,len(values)):
            if values[i] < values[i - 1]:
                values[i] = values[i - 1]       
    return values


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

def plot_model_runs(matrix, name_plot, path):
    """
        Save a boxplot of the data.
        Works only when optimization_runs is 1.

        Parameters
        ----------
        matrix: list of list of list of numbers
                or a 3D matrix
        
        name_plot : The name of the file you want to 
                    give to the plot

        path : path where the plot file is saved
    """

    plt.ioff()
    plt.xlabel('number of calls')
    plt.grid(True)
    plt.boxplot(matrix.transpose())

    if path is None:
        name_plot.split(sep=".")[0]+".png"
        plt.savefig(name_plot.split(sep=".")[0]+".png")  # save in the current working directory
    else:
        if path[-1] != '/':
            path = path + "/"
        current_dir = os.getcwd()  # current working directory
        os.chdir(path)  # change directory
        plt.savefig(name_plot)
        os.chdir(current_dir)  # reset directory to original

    plt.close()


def plot_bayesian_optimization(values, name_plot,
                               log_scale=False, path=None, conv_max=True):
    """
        Save a convergence plot of the result of a 
        Bayesian_optimization.

        Parameters
        ----------
        values : values obtained by BO

        name_plot : The name of the file you want to 
                    give to the plot

        log_scale : y log scale if True

        path : path where the plot file is saved

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
    if path is None:
        plt.savefig(name_plot.split(sep=".")[0]+".png")  # save in the current working directory
    else:
        if path[-1] != '/':
            path = path + "/"
        current_dir = os.getcwd()  # current working directory
        os.chdir(path)  # change directory
        plt.savefig(name_plot.split(sep=".")[0]+".png")
        os.chdir(current_dir)  # reset directory to original

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
    df.to_csv(name_csv.split(sep=".")[0]+".csv", index=False, na_rep='Unkown')

##############################################################################
class BestEvaluation:
    
    def __init__(self,
                 resultsBO,
                 search_space,
                 matrix_model_runs,
                 extra_metrics,
                 optimization_type):
        """
        Create an object with all the information about Bayesian Optimization
        
        """
        n_extra_metrics=matrix_model_runs.shape[0]-1
        n_calls=matrix_model_runs.shape[1]
        n_runs=matrix_model_runs.shape[2]        
 
        #Info about optimization
        self.info=dict()
        self.info.update({"number of calls":n_calls})
        self.info.update({"number of model runs":n_runs})
        self.info.update({"type_of optimization":"Maximization" if optimization_type=="Maximize" else "Minimization"})

        #Reverse the sign of minimization if the problem is a maximization    
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
        self.models_runs= dict(("iteration_"+str(i), list(matrix_model_runs[0,1,:])) for i in range(10))

        #extra metrics info
        self.extra_metrics=dict()
        j=1
        for metric in range(n_extra_metrics):
            d={metric.__class__.__name__:dict(("iteration_"+str(i), list(matrix_model_runs[j,i,:])) for i in range(n_runs))}
            self.extra_metrics.update(d)
            j=j+1
            
    def save(self,name_file):
        """
        Save results for Bayesian Optimization
        """
        with open(name_file, 'wb') as file:
          pickle.dump(self, file)

    def load(self,name):
        """
        Load results for Bayesian Optimization
        """
        with open(name, 'rb') as file:
            result=pickle.load(file)    
            
        return result