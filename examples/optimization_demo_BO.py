#import os
#os.chdir(os.path.pardir)
# Let them commented if you run this script in the main directory

from models.LDA import LDA_Model
from dataset.dataset import Dataset
from optimization.optimizer import Optimizer
from skopt.space.space import Real, Integer
from skopt import gp_minimize, forest_minimize, dummy_minimize
import multiprocessing as mp
from optimization.optimizer import default_parameters as BO_parameters
from evaluation_metrics.coherence_metrics import Coherence
import time
import resource


def simple_optimization(minimizer):
    """
        Bayesian Optimization demo. 
        This function run one optimization with the 
        selected minimizer.

        Parameters
        ----------
        minimizer : gp_minimize, forest_minimize or dummy_minimize
    """

    # Load dataset
    dataset = Dataset()
    dataset.load("preprocessed_datasets/M10/M10_lemmatized_0")
        
    # Load model
    model = LDA_Model()

    # Set model hyperparameters
    model.hyperparameters.update({ "num_topics": 25, "iterations": 200 })
    model.partitioning(False)

    # Coherence word embeddings pairwise
    parametri_metrica = {
            'texts': dataset.get_corpus(),
            'topk': 10,
            'measure': 'c_npmi'
    }
    npmi = Coherence(parametri_metrica)

    # Define optimization parameters for path (optional)
    opt_params = {}
    opt_params["minimizer"] = minimizer

    if opt_params["minimizer"] == gp_minimize:
            minimizer_stringa = "gp_minimize"
    elif opt_params["minimizer"] == dummy_minimize:
        minimizer_stringa = "random_minimize"
    elif opt_params["minimizer"] == forest_minimize:
        minimizer_stringa = "forest_minimize"
    else:
        minimizer_stringa = "None"

    path_t = "risultati/simple_"+minimizer_stringa+"/"

    # Define optimization parameters
    opt_params["n_calls"] = 5
    opt_params["n_random_starts"] = 2
    opt_params["model_runs"] = 3
    opt_params["n_jobs"] = mp.cpu_count() # Enable multiprocessing, if -1 do the same
    opt_params["save"] = False
    opt_params["save_path"] = path_t
    opt_params["early_stop"] = False
    opt_params["plot_model"]= True
    opt_params["plot_best_seen"] = True
    opt_params["plot_prefix_name"] = "plot"
    opt_params["save_models"] = True

    # Create search space for optimization
    search_space = {
        "alpha": Real(low=0.001, high=5.0),
        "eta": Real(low=0.001, high=5.0)
    }

    # Initialize optimizer
    optimizer = Optimizer(
        model,
        dataset,
        npmi,
        search_space,
        opt_params)


    # Optimize
    start_time = time.time()
    res = optimizer.optimize()
    end_time = time.time()
    total_time = end_time - start_time # Total time to optimize


    print(res.hyperparameters) # Best values for the hyperparameters
    print(res.function_values) # Score of the optimized metric
    print("Optimized metric: "+res.optimized_metric)
    print("%s seconds" % (total_time)) # Time to optimize

    stringa_parameters = str(BO_parameters) + "\nTime: " + str(total_time) + " seconds"

    res.save(name ="Result", path = path_t, parameters = stringa_parameters)

    print( resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "Kbytes" ) # RAM usage



def optimization_demo_BO():
    """
        Bayesian Optimization demo. 
        This function run 3 different optimization, one for base_minimize.
    """
    minimize_list = [
                    [dummy_minimize, "random_minimize"],
                    [forest_minimize, "forest_minimize"],
                    [gp_minimize, "gp_minimize"]
                    ]

    for minimize in minimize_list:
        print("------------------------------------------")
        print("SIMPLE OPTIMIZATION with: ", minimize[1] )
        simple_optimization(minimize[0])
        

optimization_demo_BO()

