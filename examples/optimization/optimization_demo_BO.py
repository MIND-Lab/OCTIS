# Let them commented if you run this script in the main directory
import os
os.chdir(os.path.pardir)
os.chdir(os.path.pardir)

#%% load the libraries
from models.LDA import LDA_Model
from dataset.dataset import Dataset
from optimization.optimizer import Optimizer
from skopt.space.space import Real
from evaluation_metrics.coherence_metrics import Coherence
#%% Load dataset
dataset = Dataset()
dataset.load("preprocessed_datasets/M10/M10_lemmatized_0")
    
#%% Load model
model = LDA_Model()

#%% Set model hyperparameters (not optimized by BO)
model.hyperparameters.update({ "num_topics": 25, "iterations": 200 })
model.partitioning(False)

#%% Choose of the metric function to optimize
metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
}
npmi = Coherence(metric_parameters)

#%% Create search space for optimization
search_space = {
    "alpha": Real(low=0.001, high=5.0),
    "eta": Real(low=0.001, high=5.0)
}

#%% Initialize optimizer
optimizer = Optimizer(
    model,
    dataset,
    npmi,
    search_space,
    plot_model=True,
    plot_best_seen=True,
    save_path="results/simple_GP/",
    save_name="resultsBO",
    save_models=False,
    number_of_call=6, 
    n_random_starts=3,
    optimization_type='Maximize',
    model_runs=5,
    initial_point_generator="random",   #work only for version skopt 8.0 
    surrogate_model="GP")

#%% Optimize the function npmi using Bayesian Optimization
BestObject = optimizer.optimize()

#%% Save the results to a csv
BestObject.save_to_csv("results.csv")

#%%To restart an optimization you must load the previous results
PreviousResult=BestObject.load("results/simple_GP/resultsBO.json")

#%% Initialize again the optimizer 
optimizer = Optimizer(
    model,
    dataset,
    npmi,
    search_space,
    plot_model=True,
    plot_best_seen=True,
    save_path="results/simple_GP/",
    save_name="resultsBO2",
    save_models=False,
    number_of_call=6, 
    optimization_type='Maximize',
    model_runs=2,
    surrogate_model="GP")

#%%..and launch again the optimization
BestObject2 = optimizer.restart_optimize(PreviousResult)
