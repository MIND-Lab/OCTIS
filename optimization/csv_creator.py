import pandas as pd

def save_csv(name_csv,
            dataset_name, 
            hyperparameters_name,
            metric_name,
            num_topic, 
            Surrogate,
            Acquisition,
            Time, #list of list of time
            res, 
            Maximize = False,
            time_x0 = None,):
    """
        Create a csv file to describe the topic model
        optimization.

        Parameters
        ----------
        name_csv : [string] name of the .csv file

        dataset_name: [string] name of the dataset

        hyperparameters_name : [list of string] name of the 
                            hyperparameters

        metric_name : [string] name of the metric optimized

        num_topic : number of topic

        Surrogate : [string] surrogate model used

        Acquisition : [string] acquisition function used 

        Time : [list of list] Of time for each point evaluated
            by Bayesuan_optimization()

        res : result of a Bayesian_optimization()

		Maximize : If True it will adjust the result

        time_x0 : [list of list] If not-None it adds
				x0 and y0 time to the csv file
    """    

    n_test = len( res )
    n_point = len( res[0].func_vals )

    n_row = n_test * n_point

    name_dict = {}

    if( time_x0 != None ):
        Time = time_x0 + Time


    name_dict['DATASET'] = [dataset_name]*n_row
    for hyperparameter in hyperparameters_name:
        name_dict[hyperparameter] = [] #empty
    name_dict['NUM_TOPIC'] = [num_topic]*n_row
    name_dict[ metric_name+'(optimized)'] = [] #empty
    name_dict['EXPERIMENT_ID'] = [] #empty
    name_dict['SURROGATE'] = [Surrogate]*n_row
    name_dict['ACQUISITION'] = [Acquisition]*n_row
    name_dict['NUM_ITERATION'] = [] #empty
    name_dict['TIME'] = [] #empty

    for i in range( n_point ):
        for j in range( n_test ):
            
            for k in range( len( res[j].x_iters[i] ) ): 
                name_dict[hyperparameters_name[k]].append(res[j].x_iters[i][k]) #[hyperparameters_values]

            if Maximize:
                name_dict[ metric_name+'(optimized)'].append( res[j].func_vals[i] * (-1) ) # -metric
            else:
                name_dict[ metric_name+'(optimized)'].append( res[j].func_vals[i] ) #+metric

            name_dict['EXPERIMENT_ID'].append( j )#n_test
            name_dict['NUM_ITERATION'].append( i )#n_point
            name_dict['TIME'].append(Time[i][j])
    

    df = pd.DataFrame(name_dict)


    if not(name_csv.endswith(".csv")):
        name_csv = name_csv + ".csv"

    df.to_csv(name_csv, index=False, na_rep='Unkown')

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
    if not(name_csv.endswith(".csv")):
        name_csv = name_csv + ".csv"

    df = pd.read_csv(name_csv)
    
    df[column_name] = column_data

    df.to_csv(name_csv, index=False, na_rep='Unkown')
