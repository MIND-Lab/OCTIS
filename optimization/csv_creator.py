import pandas as pd
import numpy as np
from os import path

def save_csv(name_csv,
            dataset_name, 
            hyperparameters_name,
            metric_name,
            Surrogate,
            Acquisition,
            Time, #list of list of time
            res,
            Maximize = False,
            time_x0 = None,
            len_x0 = 0):
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

        Surrogate : [string] surrogate model used

        Acquisition : [string] acquisition function used 

        Time : [list of list] Of time for each point evaluated
            by Bayesuan_optimization()

        res : result of a Bayesian_optimization()

        matrix_model_runs : The matrix of the different values
                            of the model_runs

		Maximize : If True it will adjust the result

        time_x0 : [list of list] If not-None it adds
				x0 and y0 time to the csv file
    """
    if not(name_csv.endswith(".csv")):
        name_csv = name_csv + ".csv"

    if not(name_csv.endswith(".csv")):
        name_csv_matrix = name_csv + "_matrix.csv"
    else:
        name_csv_matrix = name_csv[:-4] + "_matrix.csv"

    n_point = len( res.func_vals )
    n_row = n_point


    name_dict = {}

    if( time_x0 != None ):
        Time = time_x0 + Time


    name_dict['DATASET'] = [dataset_name]*n_row
    for hyperparameter in hyperparameters_name:
        name_dict[hyperparameter] = [] # Empty
    name_dict[ metric_name+'(optimized)'] = [] # Empty
    name_dict['SURROGATE'] = [Surrogate]*n_row
    name_dict['ACQUISITION FUNC'] = [Acquisition]*n_row
    name_dict['NUM_ITERATION'] = [] # Empty

    if( path.exists(name_csv_matrix) ):
        df_old = pd.read_csv(name_csv_matrix) 
    else:
        df_old = None
        name_dict['Mean(model_runs)'] = [None]*n_row # Unknown
        name_dict['Standard_Deviation(model_runs)'] = [None]*n_row # Unknown

    name_dict['TIME'] = [] # Empty


    for i in range( n_point ):
        for k in range( len( res.x_iters[i] ) ): 
            name_dict[hyperparameters_name[k]].append(res.x_iters[i][k]) #[hyperparameters_values]

        if Maximize:
            name_dict[ metric_name+'(optimized)'].append( res.func_vals[i] * (-1) ) # -metric
        else:
            name_dict[ metric_name+'(optimized)'].append( res.func_vals[i] ) #+metric

        name_dict['NUM_ITERATION'].append( i )#n_point
        name_dict['TIME'].append(Time[i]) #time
    
    #print("NAME DICT->",name_dict)

    df = pd.DataFrame(name_dict)
    if df_old is not None:
        if len_x0 != 0:
            #print(df_old)
            #print("MEAN",df_old.to_dict()['Mean(model_runs)'] )
            #print("STD",df_old.to_dict()['Standard_Deviation(model_runs)'] )
            dict_temp = {}
            dict_temp['Mean(model_runs)'] = [None]*len_x0
            dict_temp['Standard_Deviation(model_runs)'] = [None]*len_x0

            for element in df_old.to_dict()['Mean(model_runs)']:
                dict_temp['Mean(model_runs)'].append(element)
                #print(dict_temp['Mean(model_runs)'])
            
            for element in df_old.to_dict()['Standard_Deviation(model_runs)']:
                dict_temp['Standard_Deviation(model_runs)'].append(element)

            df_old = pd.DataFrame(dict_temp)
            #print(df_old)

        frames = [df, df_old]
        df = pd.concat(frames, axis=1, sort=False)

    #print("DataFrame->",df)

    df.to_csv(name_csv, index=False, na_rep='Unkown')

def save_matrix_csv(name_csv, matrix_model_runs):
    """
        Save a csv file named name_csv and save the
        matrix of the model runs.

        Parameters
        ----------
        name_csv : [string] name of the .csv file

        matrix_model_runs : matrix of the model runs
                            (the sub_matrix)
    """
    name_dict = {}
    sub_matrix = matrix_model_runs

    #print("sub_matrix",sub_matrix)
    media = []
    std = []
    for array in sub_matrix:
        media.append( np.mean(array) )
        std.append( np.std(array) )

    name_dict['Mean(model_runs)'] = media 
    name_dict['Standard_Deviation(model_runs)'] = std 

    df = pd.DataFrame(name_dict)
    #print("START",df)

    if not(name_csv.endswith(".csv")):
        name_csv = name_csv + "_matrix.csv"
    else:
        name_csv = name_csv[:-4] + "_matrix.csv"

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
