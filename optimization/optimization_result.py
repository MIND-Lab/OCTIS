import matplotlib.pyplot as plt
import skopt.plots

import os
import matplotlib.backends.backend_pdf
import matplotlib


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

        key_min = lambda res: res.fun
        if Maximize:
            best_values = max( optimized_result, key = key_min )
        else:
            best_values = min( optimized_result, key = key_min )

        function_values = best_values.fun
        function_solution = best_values.x


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
