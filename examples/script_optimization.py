from models.LDA import LDA_Model
from dataset.dataset import Dataset
from skopt.utils import dimensions_aslist
from evaluation_metrics.coherence_metrics import Coherence as Coherence
from optimization.optimizer import Optimizer
from optimization.optimizer_tool import dict_to_list_of_list as dict_to_list
from optimization.optimizer_tool import random_generator as random_generator
from optimization.optimizer_tool import funct_eval as funct_eval
from optimization.optimizer_tool import father_path_model as father_path_model
from optimization.optimizer import default_parameters as BO_parameters
from skopt.space.space import Real, Integer
from skopt import gp_minimize, forest_minimize, dummy_minimize
import multiprocessing as mp
from gensim.models import Word2Vec
import time
import resource
import csv
import numpy as np

#Pharameter list
cont = 0
dataset_list = ["preprocessed_datasets/M10/M10_lemmatized_0", "preprocessed_datasets/newsgroup/newsgroup_lemmatized_5"]
num_topic_list = [25, 50, 75, 100]
minimizer_list = [gp_minimize, forest_minimize]
acquisition_function_list = ["EI", "LCB"]

#dataset/modello_metrica_numerotopic_modellosurrogato_acqufunc
name_list = ["risultati/script/M10_lemmatized_0/lda_npmi_25_gp_EI/",
            "risultati/script/M10_lemmatized_0/lda_npmi_25_forest_LCB/",
            "risultati/script/M10_lemmatized_0/lda_npmi_50_gp_EI/",
            "risultati/script/M10_lemmatized_0/lda_npmi_50_forest_LCB/",
            "risultati/script/M10_lemmatized_0/lda_npmi_75_gp_EI/",
            "risultati/script/M10_lemmatized_0/lda_npmi_75_forest_LCB/",
            "risultati/script/M10_lemmatized_0/lda_npmi_100_gp_EI/",
            "risultati/script/M10_lemmatized_0/lda_npmi_100_forest_LCB/",
            "risultati/script/newsgroup_lemmatized_5/lda_npmi_25_gp_EI/",
            "risultati/script/newsgroup_lemmatized_5/lda_npmi_25_forest_LCB/",
            "risultati/script/newsgroup_lemmatized_5/lda_npmi_50_gp_EI/",
            "risultati/script/newsgroup_lemmatized_5/lda_npmi_50_forest_LCB/",
            "risultati/script/newsgroup_lemmatized_5/lda_npmi_75_gp_EI/",
            "risultati/script/newsgroup_lemmatized_5/lda_npmi_75_forest_LCB/",
            "risultati/script/newsgroup_lemmatized_5/lda_npmi_100_gp_EI/",
            "risultati/script/newsgroup_lemmatized_5/lda_npmi_100_forest_LCB/"] #should be path


#Load model LDA
model = LDA_Model()
model.partitioning(False)

# Create search space for optimization
search_space = {
    "alpha": Real(low=0.001, high=10.0),
    "eta": Real(low=0.001, high=10.0)
}

my_bounds = dict_to_list( search_space )

n_test = 30 #30
Random_points = 10 #10 Random initial points

my_x0 = random_generator(bounds = my_bounds,
                                n = Random_points, #Random points
                                n_iter = n_test)

for dataset_name in dataset_list:

    #Load dataset
    model.partitioning(False)
    dataset = Dataset()
    dataset.load(dataset_name)
    

    for numero_topic in num_topic_list:

        #Set model hyperparameters
        model.hyperparameters.update({'num_topics' : numero_topic })
        model.hyperparameters.update({'iterations' : 200}) 

        

        #Set metric npmi
        coher_par = {
        'topk':10,
        'texts': dataset.get_corpus(),
        'measure': 'c_npmi'
        }

        c_npmi = Coherence(coher_par)



        #Chose and evaluate the random POINT                      
        my_y0 = []

        # Define optimization parameters
        opt_params = {}
        opt_params["n_calls"] = 40 #40 (40 + 10 random for a total of 50)
        opt_params["minimizer"] = gp_minimize
        opt_params["different_iteration"] = n_test #30
        opt_params["n_random_starts"] = 0 
        opt_params["n_jobs"] = mp.cpu_count() # Enable multiprocessing, if -1 do the same
        opt_params["save"] = True
        opt_params["save_step"] = 1
        opt_params["save_path"] = name_list[cont] 
        opt_params["early_stop"] = False
        opt_params["plot_name"] = "gp"
        opt_params["save_name"] = "result_gp"
        opt_params["plot"] = True
        opt_params["time_x0"] = None
        opt_params['save_models'] = True

        opt_params["x0"] = my_x0
        

        # Initialize optimizer
        optimizer = Optimizer(
            model,
            dataset,
            c_npmi,
            search_space,
            opt_params)

        #y0
        my_y0 = []
        time_x0 = []
        for i in range( len(my_x0) ):
            my_y0.append([])
            time_t = []
            for j in range( len(my_x0[i]) ):
                start_time = time.time()

                path_t = father_path_model(opt_params["save_path"])
                my_y0[i].append( optimizer._objective_function( hyperparameters=  my_x0[i][j], path = path_t ) )
                
                end_time = time.time()
                total_time = end_time - start_time
                time_t.append(total_time)

            time_x0.append(time_t)

        numpy_array = np.array(time_x0)
        transpose = numpy_array.T
        time_x0 = transpose.tolist()


        opt_params["y0"] = my_y0
        opt_params["time_x0"] = time_x0

        optimizer = Optimizer(
            model,
            dataset,
            c_npmi,
            search_space,
            opt_params)

        # Disable computing of topic document matrix to optimize performance
        #optimizer.topic_document_matrix = False
        #optimizer.topic_word_matrix = False

        #print("x0 ", my_x0 )
        #print("y0 ", my_y0 )


        # Optimize
        start_time = time.time()
        res = optimizer.optimize() #gp
        end_time = time.time()
        total_time = end_time - start_time


        print(res.hyperparameters) # Best values for the hyperparameters
        print(res.function_values) # Score of the optimized metric
        print("Optimized metric: "+res.optimized_metric)
        print("%s seconds" % (total_time))

        stringa_parameters = str(BO_parameters) + "\nTime: " + str(total_time) + " seconds"
        res.save(name ="Result_gp", path = opt_params["save_path"], parameters = stringa_parameters)



        cont = cont + 1
        opt_params["minimizer"] = forest_minimize
        opt_params["plot_name"] = "forest"
        opt_params["save_path"] = name_list[cont] 
        opt_params["save_name"] = "result_forest"
        # Initialize optimizer
        optimizer = Optimizer(
            model,
            dataset,
            c_npmi,
            search_space,
            opt_params)

        # Optimize
        start_time = time.time()
        res = optimizer.optimize() #forest
        end_time = time.time()
        total_time = end_time - start_time


        print(res.hyperparameters) # Best values for the hyperparameters
        print(res.function_values) # Score of the optimized metric
        print("Optimized metric: "+res.optimized_metric)
        print("%s seconds" % (total_time))

        stringa_parameters = str(BO_parameters) + "\nTime: " + str(total_time) + " seconds"
        res.save(name ="Result_forest", path = opt_params["save_path"], parameters = stringa_parameters)

        cont = cont + 1 



 