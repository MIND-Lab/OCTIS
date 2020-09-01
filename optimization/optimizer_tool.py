import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def convergence_res(res,optimization_type="minimize"):
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
    if optimization_type=="minimize":
        for i in range(1,len(val)):
            if val[i] > val[i - 1]:
                val[i] = val[i - 1]
    else:
        for i in range(1,len(val)):
            if val[i] < val[i - 1]:
                val[i] = val[i - 1]       
    return val


def early_condition(result, n_stop, n_random):
    """
        Compute the decision to stop or not.

        Parameters
        ----------
        result : `OptimizeResult`, scipy object
                The optimization as a OptimizeResult object.
        
        n_stop : Range of points without improvement

        n_random : Random starting point

        Returns
        -------
        decision : Return True if early stop condition has been reached
    """
    n_min_len = n_stop + n_random
    
    if len(result.func_vals) >= n_min_len:
        func_vals = convergence_res(result)
        worst = func_vals[len(func_vals) - (n_stop)]
        best = func_vals[-1]
        diff = worst - best
        if diff == 0:
            return True

    return False

class Evaluation():
    """
    Representation of a single optimization iteration result
    """
    hyperparameters = {}
    function_values = {}

    def __init__(self, hyperparameters_names, hyperparameters_values,
                function_values):
        """
        Initialize class

        Parameters
        ----------
        hyperparameters_names : list of hyperparameters names
        hyperparameters_values : list of hyperparameters values
        function_values : dictionary of computed metrics values
                          key: metric name
                          value: metric value
        """

        hyperparameters = self._make_params_dict(
            hyperparameters_names,
            hyperparameters_values
        )

        self.hyperparameters = hyperparameters
        self.function_values = function_values

    def _make_params_dict(self, hyperparameters_names,
                        hyperparameters_values):
        """
        Create dictionary of hyperparameters
        from the list of name and the list of values

        Parameters
        ----------
        hyperparameters_names : list of hyperparameters names
        hyperparameters_values : list of hyperparameters values

        Returns
        -------
        params : dictionary of hyperparameters
                 key: hyperparameter name
                 value: hyperparameter value
        """
        params = {}
        for i in range(len(hyperparameters_names)):
            params[hyperparameters_names[i]] = hyperparameters_values[i]
        return params


class Best_evaluation(Evaluation):
    """
    Representation of the optimized values and each iteration
    """
    iterations = []
    optimized_result = None
    optimized_metric = None

    def __init__(self, params_names, optimized_result, Maximize, iters, optimized_metric):
        """
        Initialize class

        Parameters
        ----------
        params_names : list of hyperparameters names
        optimized_result : OptimizeResult object
        iters : list of params_values and funcction_values of each iteration
        """
        self.optimized_result = optimized_result
        iterations = []
        self.optimized_metric = optimized_metric
        function_values = {}

        function_values = optimized_result.fun #func_vals
        function_solution = optimized_result.x


        iterations.append(
            Evaluation(params_names,
                       function_solution,
                       function_values
                       )
        )
    
        super().__init__(params_names, function_solution, function_values)

        self.iterations = iterations 

    def save(self, name="save", path=None, parameters = None):
        """
        Save the values in a txt file

        Parameters
        ----------
        name : name of the txt file saved
        path : path in wich a txt with best data will be saved
        """

        if( parameters != None ):
            L  = [str(self.hyperparameters),
                "\n"+str(self.function_values),
                "\nOptimized metric: "+str(self.optimized_metric),
                "\nParameters: "+str(parameters),
                "\n------------------------------------------\n",
                str(self.optimized_result)]
        else:
            L  = [str(self.hyperparameters),
                "\n"+str(self.function_values),
                "\nOptimized metric: "+str(self.optimized_metric),
                "\n------------------------------------------\n",
                str(self.optimized_result)]

        name = name + ".txt"

        if( path == None ):
            file = open(name,"w") 
        else:
            if( path[-1] != '/' ):
                path = path + "/"
            current_dir = os.getcwd() #current working directory
            os.chdir( path ) #change directory
            file = open(name,"w") 
            os.chdir( current_dir ) #reset directory to original 
        
        file.writelines(L) 
        file.close() 
        
        
def plot_boxplot(matrix, name_plot, path):
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


def plot_bayesian_optimization(res, name_plot,
                               log_scale=False, path=None, conv_max=True):
    """
        Save a convergence plot of the result of a 
        Bayesian_optimization.

        Parameters
        ----------
        res : A Bayesian_optimization result

        name_plot : The name of the file you want to 
                    give to the plot

        log_scale : y log scale if True

        path : path where the plot file is saved

        conv_min : If True the convergence is for the min,
                    If False is for the max

    """
    if conv_max:
        #minimization problem -->maximization problem
        for j in range(len(res.func_vals)):
            res.func_vals[j] = - res.func_vals[j]
        media = convergence_res(res,optimization_type="maximize")
        xlabel='max f(x) after n calls'
    else:
        #minimization problem
        media = convergence_res(res,optimization_type="minimize")
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
