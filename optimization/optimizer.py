# Utils
from pathlib import Path  
import numpy as np
import time
# utils from skopt and sklearn
from sklearn.gaussian_process.kernels import Matern
from skopt.learning import GaussianProcessRegressor,RandomForestRegressor,ExtraTreesRegressor
from skopt          import Optimizer as skopt_optimizer
from skopt.utils    import dimensions_aslist
#utils from other files of the framework
from models.model                import save_model_output
from optimization.optimizer_tool import save_csv,early_condition,Best_evaluation
from optimization.optimizer_tool import plot_bayesian_optimization,plot_boxplot

#default kernel
Matern_kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)

# Initialize default parameters
default_parameters = {'n_calls': 100,'model_runs': 10,'n_random_starts': 10,               
                      'surrogate_model': 'RF','kernel': Matern_kernel,'acq_func': "LCB",  
                      'x0': None,'y0': None, 
                      'optimization_type': 'Maximize', 
                      'save': False,'save_models': False,'save_step': 1,'save_name': "result",'save_path': "results/",  
                      'early_stop': False, 'early_step': 10, 
                      'plot_best_seen': False,'plot_model': False,'plot_prefix_name': "B0_plot",'log_scale_plot': False,
                      'extra_metrics': [], 
                      'random_state': None}

class Optimizer:
    """
    Optimizer optimize hyperparameters to build topic models
    """

    # Values of hyperparameters and metrics for each iteration
    _iterations = []                    #counter for the BO iteration

    topk = 10                           # if False the topk words will not be computed
    topic_word_matrix = True            # if False the matrix will not be computed
    topic_document_matrix = True        # if False the matrix will not be computed

    def __init__(self, model, dataset, metric, search_space, optimization_parameters={}):
        """
        Inititalize the optimizer for the model

        Parameters
        ----------
        model : model with hyperparameters to optimize
        dataset : dataset of analysis
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

        if (optimization_parameters["save_path"][-1] != '/'):
            optimization_parameters["save_path"] = optimization_parameters["save_path"] + '/'

        self.optimization_parameters = optimization_parameters

        # Customize parameters update
        default_parameters.update(self.optimization_parameters)
        
        # Save parameters labels to use
        self.hyperparameters = list(sorted(self.search_space.keys()))

        self.extra_metrics = default_parameters["extra_metrics"]

        self.optimization_type = default_parameters['optimization_type']

        #create the directory where the results are saved
        if default_parameters["save"] == True:
            Path(default_parameters["save_path"]).mkdir(parents=True, exist_ok=True)
        
        #create of the sub-directory where the model are saved
        if default_parameters["save_models"] == True:
            model_path = default_parameters["save_path"] + "models/"
            Path(model_path).mkdir(parents=True, exist_ok=True)

        # Store the different value of the metric for each model_runs
        self.matrix_model_runs = np.zeros((default_parameters["n_calls"],default_parameters["model_runs"]))    
        
        # Store the different values of the extra metric for each model runs
        self.matrix_model_runs_extra_metrics = np.zeros((len(default_parameters["extra_metrics"]),default_parameters["n_calls"],default_parameters["model_runs"]))

    def _objective_function(self, hyperparameters):
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
            #Score of the model 
            score = self.metric.score(model_output)
            
            different_model_runs.append(score)
            self.matrix_model_runs[self.current_call, i] = score
            
            #Update of the extra metric values
            j=0
            for extra_metric in self.extra_metrics:
                self.matrix_model_runs_extra_metrics[j,self.current_call, i]= extra_metric.score(model_output)
                j=j+1
            
            # Save the model for each run
            if default_parameters["save_models"]:
                name = str(self.current_call) + "_" + str(i) 
                save_model_path = default_parameters["save_path"] + "models/" + name
                save_model_output(model_output, save_model_path)
                
        #the output for BO is the median over different_model_runs 
        result = np.median(different_model_runs)
        
        # Save iteration data
        metrics_values = {self.metric.__class__.__name__: result}
        iteration = [hyperparameters, metrics_values] 
        self._iterations.append(iteration)

        if self.optimization_type == 'Maximize':
            result = - result

        # Update evaluation of objective function
        self.current_call = self.current_call + 1

        #Boxplot for matrix_model_runs
        if default_parameters["plot_model"]:
            plot_boxplot(self.matrix_model_runs[:self.current_call,:], 
                         default_parameters["plot_prefix_name"].split(".")[0]+"_model_runs_"+self.metric.info()["name"], 
                         default_parameters["save_path"])
            #Boxplot of extrametrics (if any)
            j=0
            for extra_metric in self.extra_metrics:
                 plot_boxplot(self.matrix_model_runs_extra_metrics[j,:self.current_call,:], 
                             default_parameters["plot_prefix_name"].split(".")[0]+"_model_runs_"+extra_metric.info()["name"], 
                             default_parameters["save_path"])  
                 j=j+1
                 
        return result

    def Bayesian_optimization(self, f,  # = self.self._objective_function,#
                              bounds,  # = params_space_list,#
                              number_of_call=default_parameters["n_calls"],
                              n_random_starts=default_parameters["n_random_starts"],
                              model_runs=default_parameters["model_runs"],
                              surrogate_model=default_parameters["surrogate_model"],
                              kernel=default_parameters["kernel"],
                              acq_func=default_parameters["acq_func"],
                              random_state=default_parameters["random_state"],
                              x0=default_parameters["x0"],
                              y0=default_parameters["y0"],
                              save=default_parameters["save"],
                              save_step=default_parameters["save_step"],
                              save_name=default_parameters["save_name"],
                              save_path=default_parameters["save_path"],
                              early_stop=default_parameters["early_stop"],
                              early_step=default_parameters["early_step"],
                              plot_best_seen=default_parameters["plot_best_seen"],
                              plot_prefix_name=default_parameters["plot_prefix_name"],
                              log_scale_plot=default_parameters["log_scale_plot"]):
        """
            Bayesian_optimization

            Input parameters
            ----------
            f : Function to minimize. 

            bounds : List of search space dimensions. Each search dimension can be defined either as
                    - (lower_bound, upper_bound) tuple (for Real or Integer dimensions),
                    - (lower_bound, upper_bound, "prior") tuple (for Real dimensions),
                    - list of categories (for Categorical dimensions), or
                    - instance of a Dimension object (Real, Integer or Categorical).

            number_of_call : Number of calls to f 

            n_random_starts : Number of evaluations of f with random points before surrogato model is used

            model_runs: Number of different evaluation of the function in the same point
                        and with the same hyperparameters. Usefull with a lot of noise.
            
            surrogate_model : The regressor to use as surrogate model. Can be either
                                    - "GP" for gaussian process regressor
                                    - "RF" for random forest regressor
                                    - "ET" for extra trees regressor

            kernel : The kernel specifying the covariance function of the GP.
            
            acq_func : Function to minimize over the minimizer prior. Can be either
                        - "LCB" for lower confidence bound.
                        - "EI" for negative expected improvement.
                        - "PI" for negative probability of improvement.
                        - "EIps" for negated expected improvement.
                        - "PIps" for negated probability of improvement.
            
            random_state : Set random state to something other than None for reproducible results.
            
            x0 : Initial input points.
            
            y0 : Evaluation of initial input points.
                     
            save : [boolean] Save the Bayesian Optimization in a .pkl and .cvs file 
            
            save_step : Integer interval after which save the .pkl file
            
            save_name : Name of the .pkl and .cvs file saved.
            
            save_path : Path where .pkl, plot and result will be saved.
            
            early_stop : [boolean] Early stop policy.
                        It will stop an interaction if it doesn't
                        improve for early_step evaluations.
            
            early_step : Integer interval after which a current optimization run
                        is stopped if it doesn't improve.
            
            plot_best_seen : [boolean] Plot the convergence of the Bayesian optimization.
                    If save is True the plot is update every save_step evaluations.
            
            plot_prefix_name : Prefix of the name of the .png file where the plots are saved.
            
            log_scale_plot : [boolean] If True the "y_axis" of the plot
                            is set to log_scale
            
            n_points : Number of points to sample to determine the next “best” point.        
            
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
        
        if save and save_path is not None:
            Path(save_path).mkdir(parents=True, exist_ok=True)
        
        #### Choice of the surrogate model
        # Random forest
        if surrogate_model == "RF":
            estimator=RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
            surrogate_model_name = "Random forest regressor"
        # Extra Tree
        elif surrogate_model == "ET":
            estimator=ExtraTreesRegressor(n_estimators=100,min_samples_leaf=3)
            surrogate_model_name = "Extra tree regressor"            
        # GP Minimize
        elif surrogate_model == "GP":
            estimator = GaussianProcessRegressor(kernel=kernel,random_state=random_state)
            surrogate_model_name = "Gaussian process regressor" 
        # Random Search
        elif surrogate_model == "RS":
            estimator="dummy"
            surrogate_model_name = "Random search (no surrogate model)" 
        else:
             print("Error: surrogate_model does not exist ")
             return None           
        #print of information about the optimization
        Optimizer.print_info_optimization(default_parameters,surrogate_model_name)
       
        #Creation of a general skopt optimizer
        opt = skopt_optimizer(bounds, base_estimator=estimator, 
                              acq_func=acq_func,
                              n_random_starts=n_random_starts, 
                              n_initial_points=n_random_starts,
                              acq_optimizer="sampling", 
                              acq_optimizer_kwargs={"n_points": 10000, "n_restarts_optimizer": 5,"n_jobs": 1},
                              acq_func_kwargs={"xi": 0.01, "kappa": 1.96},
                              random_state=random_state)
              
        time_eval = []
        #time_eval_BO = []
        ####First, x0 and y0 (is any) are included in the surrogate model
        lenx0=0#len(x0)
        
        if x0 is not None:
            if y0 is not None:
                res=opt.tell(x0,y0,fit=True)
                time_eval.append([0]*len(y0)) 
            else:
                #The values of y0 must be computed
                for i in x0:
                    start_time = time.time()

                    f_val = f(i)
                    res=opt.tell(i, f_val) 
        
                    end_time = time.time()
                    total_time = end_time - start_time
                    time_eval.append(total_time)                     
        
        number_of_call_r=number_of_call-lenx0

        if number_of_call_r <= 0:
              print("Error: number_of_call is less then len(x0)")
              return None

        ####for loop to perform Bayesian Optimization        
        for i in range(number_of_call_r):          

            start_time = time.time()  
            next_x = opt.ask()                              #next point proposed by BO
            #end_time=time.time()
            #total_time_BO = end_time - start_time          #computational time for BO to propose the next point 
            #time_eval_BO.append(total_time_BO)            
            #start_time = time.time()
            f_val = f(next_x)                               #evaluation of the objective function for next_x
            res=opt.tell(next_x, f_val)                     #update of the opt using (next_x,f_val)
            
            end_time = time.time()
            total_time_function = end_time - start_time     #computational time for next_x (BO+Function evaluation)
            time_eval.append(total_time_function)

            if save and i % save_step == 0:
                save_csv(name_csv=save_path+save_name,
                         res=res,
                         matrix_model_runs=self.matrix_model_runs[:self.current_call,:], 
                         matrix_model_runs_extra_metrics=self.matrix_model_runs_extra_metrics[:,:self.current_call,:],
                         extra_metrics=self.extra_metrics,
                         dataset_name=self.dataset.get_metadata()["info"]["name"],
                         hyperparameters_name=self.hyperparameters,
                         metric_name=self.metric.__class__.__name__,
                         surrogate_model_name=surrogate_model_name, 
                         acquisition_function_name=acq_func,
                         times=time_eval)    
                
                if plot_best_seen:  ####secondo me si può farlo salvare sempre invece
                    plot_bayesian_optimization(res, plot_prefix_name.split(".")[0]+"_best_seen", log_scale_plot,
                                                    path=save_path,conv_max=self.optimization_type == 'Maximize')

            if early_stop and  early_condition(res, early_step, n_random_starts):
                print("Stop because of early stopping condition")
                break  
        
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

        self.extra_metrics = default_parameters["extra_metrics"]

        self.optimization_type = default_parameters['optimization_type']

        # Optimization call
        optimize_result = self.Bayesian_optimization(f=self._objective_function,
                                       bounds=params_space_list,
                                       number_of_call=default_parameters["n_calls"],
                                       surrogate_model=default_parameters["surrogate_model"],
                                       kernel=default_parameters["kernel"],
                                       acq_func=default_parameters["acq_func"],                                    
                                       random_state=default_parameters["random_state"],
                                       x0=default_parameters["x0"], y0=default_parameters["y0"],
                                       n_random_starts=default_parameters["n_random_starts"],
                                       save=default_parameters["save"],
                                       save_step=default_parameters["save_step"],
                                       save_name=default_parameters["save_name"],
                                       save_path=default_parameters["save_path"],
                                       early_stop=default_parameters["early_stop"],
                                       early_step=default_parameters["early_step"],
                                       plot_best_seen=default_parameters["plot_best_seen"],
                                       plot_prefix_name=default_parameters["plot_prefix_name"],
                                       log_scale_plot=default_parameters["log_scale_plot"])

        # Create Best_evaluation object from optimization results
        result = Best_evaluation(self.hyperparameters,
                                 optimize_result,
                                 self.optimization_type == 'Maximize',
                                 self._iterations,
                                 self.metric.__class__.__name__)

        return result

    def print_info_optimization(default_parameters,surrogate_model_name):
        
        print("------------------------------------------")
        print("Bayesian optimization parameters:\n-n_calls: ", default_parameters["n_calls"],
              "\n-model_runs: ", default_parameters["model_runs"],
              "\n-n_random_starts: ", default_parameters["n_random_starts"],
              "\n-minimizer: ", surrogate_model_name)
        if default_parameters["surrogate_model"] in ["GP","RF","ET"]:
            print("-acq_func: ", default_parameters["acq_func"])
        if default_parameters["surrogate_model"] == "GP":
            print("-kernel: ", default_parameters["kernel"])
        print("------------------------------------------")       