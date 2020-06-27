import csv
import resource
from optimization.optimizer_tool import print_func_vals as print_func_vals
from optimization.optimizer_tool import print_x_iters as print_x_iters

#what if X0 and Y0?

def support_csv(name_csv, hyperparameters_name, data ): 
    """
        Support function for save_csv.
        Open the csv file and store the data.

        Parameters
        ----------
        name_csv : [string] name of the .csv file
        hyperparameters_name : [list of string] name of the 
                            hyperparameters
        data : [list of list] where the data are stored
    """

    csvfile = open(name_csv, 'w', newline='', encoding='utf-8')

    fieldnames = ['DATASET', 
                'NUM_TOPIC', 
                'NPMI', 
                'TOPIC_DIVERSITY', 
                'KL_B', 
                'KL_U', 
                'KL_V', 
                'EXPERIMENT_ID', 
                'SURROGATE',
                'ACQUISITION', 
                'NUM_ITERATION', 
                'TIME' ]

    #Add the hyperparameters
    for i in range( len(hyperparameters_name) ):
        fieldnames.insert(i + 1, hyperparameters_name[i])

    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(fieldnames)

    for single_row in data:
        writer.writerow(single_row)


def save_csv(name_csv,
            dataset_name, 
            hyperparameters_name, 
            num_topic, 
            Surrogate,
            Acquisition,
            Time, #list of list of time
            res,
            Maximize = False,
            time_x0 = None,
            topic_diversity = None, 
            KL_B = None, 
            KL_U = None, 
            KL_V = None):
    """
        Create a csv file to describe the topic model
        optimization. 
        Parse the data for support_csv

        Parameters
        ----------
        name_csv : [string] name of the .csv file
        dataset_name: [string] name of the dataset
        hyperparameters_name : [list of string] name of the 
                            hyperparameters
        num_topic : number of topic
        Surrogate : [string] surrogate model used
        Acquisition : [string] acquisition function used 
        Time : [list of list] Of time for each point evaluated
            by Bayesuan_optimization()
        res : result of a Bayesian_optimization()
        topic_diversity : If not None is added 
        KL_B : If not None is added  
        KL_U : If not None is added  
        KL_V : If not None is added 
    """
    data = []

    n_test = len( res )
    n_point = len( res[0].func_vals ) 



    if( time_x0 != None ):
        #print("Time_x0 ", time_x0 )
        #print("time", Time)
        #for i in time_x0:
        Time = time_x0 + Time

    for i in range( n_point ):
        for j in range( n_test ):
            data_t = []
            data_t = [dataset_name]
            
            for k in range( len( res[j].x_iters[i] ) ): 
                data_t.append(res[j].x_iters[i][k]) #[hyperparameters_values]

            data_t.append(num_topic)
            if Maximize:
                data_t.append( res[j].func_vals[i] * (-1) ) # -npmi
            else:
                data_t.append( res[j].func_vals[i] ) #npmi
            data_t.append( j )#n_test
            data_t.append(Surrogate)
            data_t.append(Acquisition)
            data_t.append( i )#n_point
            #print("i ", i)
            #print("j ", j)
            #print_x_iters(res)
            #print_func_vals(res)
            #print("time ", Time )
            data_t.append(Time[i][j])


            if(KL_V == None ):
                data_t.insert( len(hyperparameters_name) + 3, '-')
            else:
                data_t.insert( len(hyperparameters_name) + 3, KL_V[i][j] )
            
            if(KL_U == None ):
                data_t.insert( len(hyperparameters_name) + 3, '-')
            else:
                data_t.insert( len(hyperparameters_name) + 3, KL_U[i][j] )

            if(KL_B == None ): 
                data_t.insert( len(hyperparameters_name) + 3, '-')
            else:
                data_t.insert( len(hyperparameters_name) + 3, KL_B[i][j] )

            if( topic_diversity == None ):
                data_t.insert( len(hyperparameters_name) + 3, '-')
            else:
                data_t.insert( len(hyperparameters_name) + 3, topic_diversity[i][j] )

            data.append( data_t )
                
    support_csv(name_csv, hyperparameters_name, data)




#print(resource.getpagesize() )
#print( resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "kilobytes" )
    

