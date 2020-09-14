# Let them commented if you run this script in the main directory
import os
os.chdir(os.path.pardir)
os.chdir(os.path.pardir)


from models.LDA import LDA_Model
from dataset.dataset import Dataset
from optimization.optimizer import Optimizer
from skopt.space.space import Real
from evaluation_metrics.coherence_metrics import Coherence
import time


# Load dataset
dataset = Dataset()
dataset.load("preprocessed_datasets/M10/M10_lemmatized_0")
    
# Load model
model = LDA_Model()

# Set model hyperparameters (not optimized by BO)
model.hyperparameters.update({ "num_topics": 25, "iterations": 200 })
model.partitioning(False)

# Choose of the metric function to optimize
metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
}
npmi = Coherence(metric_parameters)

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
    plot_model=True,
    plot_best_seen=True,
    save_path="results/simple_RF/",
    save_name="resultsBO",
    save_csv=True,
    save_models=False,
    number_of_call=10, 
    n_random_starts=3,
    optimization_type='Maximize',
    model_runs=5,
    initial_point_generator="random",   #work only for version skopt 8.0 
    surrogate_model="RF")

# Optimize the function npmi using Bayesian Optimization
start_time = time.time()
BestObject = optimizer.optimize()
end_time = time.time()
total_time = end_time - start_time # Total time to optimize

#The results of the optimization are saved in the object BestObject

#To start a new optimization from the previous results we need to include x0 and y0 in the object optimizer...


# import pandas as pd

# optimizer = Optimizer(
#     model,
#     dataset,
#     npmi,
#     search_space,
#     plot_model=True,
#     plot_best_seen=True,
#     save_path="results/simple_RF/",
#     save_name="resultsBO",
#     save_csv=True,
#     save_models=True,
#     number_of_call=10, 
#     n_random_starts=3,
#     optimization_type='Maximize',
#     model_runs=3,
#     surrogate_model="RF",
#     x0=pd.DataFrame(BestObject.x_iters_as_dict).values.tolist(),   #!!!!!
#     y0=BestObject.func_vals)

# # #..and launch again the optimize
# start_time = time.time()
# BestObject = optimizer.optimize()
# end_time = time.time()
total_time = end_time - start_time # Total time to optimize