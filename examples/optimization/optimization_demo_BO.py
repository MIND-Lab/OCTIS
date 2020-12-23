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
dataset.load("preprocessed_datasets/m10/M10_lemmatized_0")
    
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
#%% Optimize the function npmi using Bayesian Optimization
optimizer=Optimizer()
BestObject = optimizer.optimize(model,
    dataset,
    npmi,
    search_space,
    plot_model=False,
    plot_best_seen=False,
    save_path="results/simple_GP/",
    save_name="resultsBO",
    save_models=False,
    number_of_call=5, 
    n_random_starts=3,
    optimization_type='Maximize',
    model_runs=2,
    initial_point_generator="lhs",   #work only for version skopt 8.0 
    surrogate_model="GP")

#%% Save the results to a csv
BestObject.save_to_csv("results.csv")

#%% Resume the optimization
path=BestObject.name_json
optimizer=Optimizer()
optimizer.resume_optimization(path,extra_evaluations=3)
