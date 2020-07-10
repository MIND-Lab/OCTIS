import csv
#from optimization.optimizer_tool import print_func_vals as print_func_vals
#from optimization.optimizer_tool import print_x_iters as print_x_iters


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

		Maximize : If True it will adjust the result

        time_x0 : [list of list] If not-None it adds
				x0 and y0 time to the csv file

        topic_diversity : If not None is added 

        KL_B : If not None is added  

        KL_U : If not None is added  

        KL_V : If not None is added 

    """
    data = []

    n_test = len( res )
    n_point = len( res[0].func_vals ) 



    if( time_x0 != None ):
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
    

def search_extra_csv(name_csv):
    """
        Search for "-" in the second row.
        It needs for upload_csv to know where
        to write.

        Parameters
        ----------
        name_csv : [string] name of the .csv file
    """
    csvfile_r = open(name_csv, 'r', newline='')
    read = csv.reader(csvfile_r)

    search_list = []
    flag = False
    for row in read:
        cont = 0
        for element in row:
            if( element == '-' ):
                search_list.append(cont)
            cont = cont + 1
        if(flag):
            break
        flag = True

    return search_list


def upload_csv(name_csv,
            topic_diversity, 
            KL_B, 
            KL_U, 
            KL_V):
    """
        Upload save_csv.
        Ad the data of topic_diversity, KL_B, KL_U, KL_V
        metrics.

        Parameters
        ----------
        name_csv : [string] name of the .csv file

        topic_diversity : [list] where topic_diversity data are stored 

        KL_B : [list] where KL_B data are stored  

        KL_U : [list] where KL_U data are stored  

        KL_V : [list] where KL_V data are stored
    """

    search_list = search_extra_csv(name_csv)
    if( len(search_list) < 4 ):
        return

    csvfile_r = open(name_csv, 'r', newline='')
    read = csv.reader(csvfile_r)

    save_csv = [] #list of list
    for row in read:
        save_csv.append(row)

    flag = False
    cont = 0
    for row in save_csv:
        if( flag ):
            row[search_list[0]] = topic_diversity[cont]
            row[search_list[1]] = KL_B[cont]
            row[search_list[2]] = KL_U[cont]
            row[search_list[3]] = KL_V[cont]
            cont = cont + 1
        flag = True


    
    csvfile_w = open(name_csv, 'w', newline='')
    writer = csv.writer(csvfile_w, quoting=csv.QUOTE_ALL)

    for row in save_csv:
        writer.writerow(row)


def add_column(name_csv,
            column_name,
            column_data):
    """
        Upload name_csv.
        Append the data column_data.

        Parameters
        ----------
        name_csv : [string] name of the .csv file

        column_name : [string] name of the column you want to add

        column_data : [list] list of value you want to add in the last column.
                        MUST be long as the other columns.
    """

    csvfile_r = open(name_csv, 'r', newline='')
    read = csv.reader(csvfile_r)

    save_csv = [] #list of list
    for row in read:
        save_csv.append(row)


    flag = False
    cont = 0
    for row in save_csv:
        if( flag ):
            row.append(column_data[cont])
            cont = cont + 1
        else:
            row.append(column_name)
        flag = True



    
    csvfile_w = open(name_csv, 'w', newline='')
    writer = csv.writer(csvfile_w, quoting=csv.QUOTE_ALL)

    for row in save_csv:
        writer.writerow(row)
