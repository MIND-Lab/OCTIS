#import os
#os.chdir(os.path.pardir)

from models.LDA import LDA_Model
from dataset.dataset import Dataset
from evaluation_metrics.coherence_metrics import Coherence_word_embeddings_pairwise, Coherence_word_embeddings_centroid
from evaluation_metrics.topic_significance_metrics import KL_uniform
from optimization.optimizer import Optimizer
from optimization.optimizer import default_parameters as BO_parameters
from skopt.space.space import Real, Integer
from skopt import gp_minimize, forest_minimize, dummy_minimize
import multiprocessing as mp
from gensim.models import Word2Vec
import time
    
# Load dataset
dataset = Dataset()
dataset.load("preprocessed_datasets/newsgroup/newsgroup_lemmatized_10")
    
# Load model
model = LDA_Model()

# Set model hyperparameters
model.hyperparameters.update({'num_topics':20})

# Coherence word embeddings pairwise
metric_params = {
    'topk':10,
    'w2v_model': Word2Vec(dataset.get_corpus())
}
c_we_p = Coherence_word_embeddings_pairwise(metric_params)

# Coherence word embeddings pairwise
c_we_c = Coherence_word_embeddings_centroid(metric_params)

# Define optimization parameters
opt_params = {}
opt_params["n_calls"] = 40
opt_params["minimizer"] = gp_minimize
opt_params["different_iteration"] = 5
opt_params["n_random_starts"] = 4 
opt_params["extra_metrics"] = [c_we_p] # List of extra metrics
opt_params["n_jobs"] = mp.cpu_count() # Enable multiprocessing, if -1 do the same
opt_params["save"] = True
opt_params["save_step"] = 1
opt_params["save_path"] = None #"risultati/giacomo/" 
opt_params["early_stop"] = True
opt_params["plot"] = True

# Create search space for optimization
search_space = {
    "alpha": Real(low=0.001, high=5.0),
    "eta": Real(low=0.001, high=5.0)
}

# Initialize optimizer
optimizer = Optimizer(
    model,
    dataset,
    c_we_c,
    search_space,
    opt_params)

# Disable computing of topic document matrix to optimize performance
optimizer.topic_document_matrix = False
optimizer.topic_word_matrix = False

# Optimize
start_time = time.time()
res = optimizer.optimize()
end_time = time.time()
total_time = end_time - start_time


print(res.hyperparameters) # Best values for the hyperparameters
print(res.function_values) # Score of the optimized metric
print("Optimized metric: "+res.optimized_metric)
print("%s seconds" % (total_time))

stringa_parameters = str(BO_parameters) + "\nTime: " + str(total_time) + " seconds"

res.save(name ="Result", path = "risultati/giacomo/", parameters = stringa_parameters)

#res.plot