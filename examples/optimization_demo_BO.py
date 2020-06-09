#import os
#os.chdir(os.path.pardir)

from models.LDA import LDA_Model
from dataset.dataset import Dataset
from evaluation_metrics.diversity_metrics import Topic_diversity
from evaluation_metrics.topic_significance_metrics import KL_uniform
from optimization.optimizer import Optimizer
from skopt.space.space import Real, Integer
from skopt import gp_minimize, forest_minimize, dummy_minimize
import multiprocessing as mp
from gensim.models import Word2Vec

# Load dataset
dataset = Dataset()
dataset.load("preprocessed_datasets/newsgroup/newsgroup_lemmatized_10")

# Load model
model = LDA_Model()

# Set model hyperparameters
model.hyperparameters['num_topics'] = 20

# Define metrics
topic_diversity = Topic_diversity()
kl_uniform = KL_uniform()

# Define optimization parameters
opt_params = {}
opt_params["n_calls"] = 30
opt_params["minimizer"] = forest_minimize
opt_params["different_iteration"] = 3
opt_params["n_random_starts"] = 5
opt_params["extra_metrics"] = [kl_uniform] # List of extra metrics
opt_params["n_jobs"] = mp.cpu_count() -1 # Enable multiprocessing
opt_params["verbose"] = True

# Create search space for optimization
search_space = {
    "alpha": Real(low=0.001, high=5.0),
    "eta": Real(low=0.001, high=5.0)
}

# Initialize optimizer
optimizer = Optimizer(
    model,
    dataset,
    topic_diversity,
    search_space,
    opt_params)

# Disable computing of topic document matrix to optimize performance
optimizer.topic_document_matrix = False

# Optimize
res = optimizer.optimize()

print(res.hyperparameters) # Best values for the hyperparameters
print(res.function_values) # Score of the optimized metric
print("Optimized metric: "+res.optimized_metric)

res.save(name ="Result", path = "optimization")
    

# Plot data NOT WORKING
#res.plot_all(metric="Coherence_word_embeddings_pairwise")