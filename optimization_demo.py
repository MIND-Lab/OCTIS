from models.NMF import NMF_Model
from dataset.dataset import Dataset
from evaluation_metrics.diversity_metrics import Topic_diversity
from optimization.optimizer import Optimizer
from skopt.space.space import Real, Integer
import multiprocessing as mp


# Load dataset
dataset = Dataset()
dataset.load("preprocessed_datasets/newsgroup/newsgroup_lemmatized_10")


model = NMF_Model(dataset)


# Create search space for optimization
search_space = {
    "num_topics": Integer(low=18, high=22),
    "alpha": Real(low=0.1, high=2.0),
    "eta": Real(low=0.1, high=2.0)
}

# Define optimization parameters
opt_params = {}
opt_params["n_calls"] = 10
opt_params["n_random_starts"] = 2
opt_params["n_jobs"] = mp.cpu_count() # Enable multiprocessing


# Initialize optimizer
optimizer = Optimizer(model, Topic_diversity, search_space, {'topk': 10}, opt_params)

# Disable computing of topic document matrix and topic word matrix
# To optimize code
optimizer.topic_document_matrix = False
optimizer.topic_word_matrix = False



# Optimize
res = optimizer = optimizer.optimize()

print(res[0]) # Best values for the hyperparameters
print(res[1].fun) # Score of the metric with the best hyperparameters