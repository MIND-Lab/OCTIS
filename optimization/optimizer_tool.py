import inspect
import os
import re
import statistics

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import distance as dist_eu
from skopt import dump, load


def get_concat_h(im1, im2):
    """
        Concat two images as it follows:
        -    im1 = Image.open('Comparing Acquisition Function Mean.png')
        -    im2 = Image.open('Comparing Acquisition Function Mean 1x.png')
        -    get_concat_h(im1, im2).save('h.jpg')

        -PIL.Image module needed

        Parameters
        ----------
        im1 : First image

        im2 : Second image 

        Returns
        -------
        dst : im1 and im2 concatenation
            
    """
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def dict_to_list_of_list(a_dict):
    """
        Return a list of a given dictionary.

        Parameters
        ----------
        a_dict : a dictionary

        Returns
        -------
        list_of_list : A list of the values
                       of dict
            
    """
    list_of_list = []
    for element in a_dict:
        list_of_list.append(a_dict[element].bounds)
    return list_of_list


def list_to_dict(a_list, a_dictionary):
    """
        Return a dictionary of a given list.
        The key of the dictionary are the same of dict.

        Parameters
        ----------
        a_list : list

        a_dictionary : a dictionary

        Returns
        -------
        space : A dictionary with the value of list and the
                key of dict
            
    """
    space = {}
    i = 0
    for element in a_dictionary:
        space[element] = a_list[i]
        i = i + 1
    return space


def random_generator(bounds, n, random_state=None):
    """
        Return a list of n random numbers in the bounds.
        Random numbers are generated with 
        uniform distribution.

        Parameters
        ----------
        bounds : A dict of bound for the random numbers

        n : Number of random numbers

        random_state : The random state

        Returns
        -------
        array : A list of n random numbers 
                in the bounds
    """
    array = []
    for i in range(n):
        array.append([])

    for i in range(n):
        for b in bounds:
            # print( bounds[b] )
            array[i].append(bounds[b].rvs(n_samples=1, random_state=random_state)[0])
    return array


def funct_eval(funct, points):
    """
        Return a list of the evaluation of the points 
        in the function funct
        Build to work with random_generator()

        Parameters
        ----------
        funct : A function the return a single value

        points : A list of point

        Returns
        -------
        array : A list of evaluation
    """
    array = []
    for i in range(len(points)):
        array.append([])
    for i in range(len(points)):
        for j in range(len(points[0])):
            array[i].append(funct(points[i][j]))
    return array


def convergence_res(res):
    """
        Given a single element of a
        Bayesian_optimization return the 
        convergence of y

        Parameters
        ----------
        res : A single element of a 
            Bayesian_optimization result

        Returns
        -------
        val : A list with the best min seen for 
            each evaluation
    """
    # print("RES", res)
    val = res.func_vals
    for i in range(len(val)):
        if i != 0 and val[i] > val[i - 1]:
            val[i] = val[i - 1]
    return val


def convergence_res_max(res):
    """
        Given a single element of a
        Bayesian_optimization return the 
        convergence of y but max

        Parameters
        ----------
        res : A single element of a 
            Bayesian_optimization result

        Returns
        -------
        val : A list with the best min seen for 
            each evaluation
    """
    val = res.func_vals
    for i in range(len(val)):
        if (i != 0 and val[i] < val[i - 1]):
            val[i] = val[i - 1]
    return val


def early_condition(result, n_stop, n_random):
    """
        Compute the decision to stop or not.

        Parameters
        ----------
        result : `OptimizeResult`, scipy object
                The optimization as a OptimizeResult object.
        
        n_stop : Range of points without improvement

        n_random : Random starting point

        Returns
        -------
        decision : Return True if early stop condition has been reached
    """
    n_min_len = n_stop + n_random
    if len(result.func_vals) >= n_min_len:
        func_vals = convergence_res(result)
        worst = func_vals[len(func_vals) - (n_stop)]
        best = func_vals[-1]
        diff = worst - best
        if diff == 0:
            return True

    return False


def iteration_without_improvement(result):
    """
        Compute the decision to stop or not.

        Parameters
        ----------
        result : `OptimizeResult`, scipy object
                The optimization as a OptimizeResult object.
        
        n_stop : Range of points without improvement

        n_random : Random starting point

        Returns
        -------
        decision : Return True if early stop condition has been reached
    """
    cont = 0
    if len(result) > 0:
        func_vals = convergence_res(result)
        last = func_vals[-1]
        cont = func_vals.count(last)

    return cont


def varname(p):
    """
        Return the name of the variabile p
        -inspect module needed
        -re module needed

        Parameters
        ----------
        p : variable with a name

        Returns
        -------
        m : Name of the variable p
            
    """
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def median_number(lista):
    """
        Given a list it 
        return the median 

        Parameters
        ----------
        lista : [list] List of numbers

        Returns
        -------
        val : The median of the numbers
    """
    val = statistics.median(lista)
    return val


def median(list_of_res):
    """
        Given a Bayesian_optimization result 
        the median of the min y found
        -statistics module needed

        Parameters
        ----------
        list_of_res : A Bayesian_optimization result

        Returns
        -------
        val : The median of the min y found
    """
    r = []
    for res in list_of_res:
        r.append(list(convergence_res(res)))
    val = []
    for i in r:
        val.append(i[-1])
    val = statistics.median(val)
    return val


def len_func_vals(list_of_res):
    """
        Given a Bayesian_optimization result 
        return a list of func_vals lenght

        Parameters
        ----------
        list_of_res : A Bayesian_optimization result

        Returns
        -------
        lista : A list of the lenght with of the
                func_vals
    """
    lista = []
    for i in list_of_res:
        lista.append(len(i.func_vals))
    return lista


def total_mean_max(list_of_res):
    """
        Given a Bayesian_optimization result 
        return a list of the mean with the other 
        tests runned but for the max

        Parameters
        ----------
        list_of_res : A Bayesian_optimization result

        Returns
        -------
        media : A list of the mean with the other 
                tests runned
    """
    r = []
    different_iteration = len(list_of_res)
    for res in list_of_res:
        r.append(list(convergence_res_max(res)))
    a = []
    media = []
    max_len = max(len_func_vals(list_of_res))
    for i in range(max_len):
        for j in range(different_iteration):
            if (len(r[j]) > i):
                a.append(r[j][i])
        media.append(np.mean(a, dtype=np.float64))
        a = []
    return media


def total_mean(list_of_res):
    """
        Given a Bayesian_optimization result 
        return a list of the mean with the other 
        tests runned 

        Parameters
        ----------
        list_of_res : A Bayesian_optimization result

        Returns
        -------
        media : A list of the mean with the other 
                tests runned
    """
    r = []
    different_iteration = len(list_of_res)
    for res in list_of_res:
        r.append(list(convergence_res(res)))
    a = []
    media = []
    max_len = max(len_func_vals(list_of_res))
    for i in range(max_len):
        for j in range(different_iteration):
            if len(r[j]) > i:
                a.append(r[j][i])
        media.append(np.mean(a, dtype=np.float64))
        a = []
    return media


def convergence_res_x(res, r_min):
    """
        Given a single element of a
        Bayesian_optimization and the argmin
        of the function return the convergence of x
        centred around the lowest distance 
        from the argmin
        -scipy.spatial.distance module needed


        Parameters
        ----------
        res : A single element of a 
            Bayesian_optimization result

        min : the argmin of the function in form 
            of a list as it follows:
            -[[-pi, 12.275], [pi, 2.275], [9.42478, 2.475] ]

        Returns
        -------
        distance : A list with the distance between 
            the best x seen for each evaluation
            and the argmin
    """
    val = res.x_iters
    distance = []
    if len(r_min) == 1:
        for i in range(len(val)):
            if i != 0 and dist_eu.euclidean(val[i], r_min) > distance[i - 1]:
                distance.append(distance[i - 1])
            else:
                distance.append(dist_eu.euclidean(val[i], r_min))
        return distance
    else:
        distance_all_min = []
        for i in range(len(val)):
            for j in range(len(r_min)):
                distance_all_min.append(dist_eu.euclidean(val[i], r_min[j]))
            min_distance = min(distance_all_min)
            if (i != 0 and min_distance > distance[i - 1]):
                distance.append(distance[i - 1])
            else:
                distance.append(min_distance)
            distance_all_min = []
        return distance


def total_mean_x(list_of_res, min):
    """
        Given a Bayesian_optimization result
        and the argmin of the function return 
        the mean of x centred around the lowest 
        distance from the argmin

        Parameters
        ----------
        res : A Bayesian_optimization result

        min : the argmin of the function in form 
            of a list as it follows:
            -[[-pi, 12.275], [pi, 2.275], [9.42478, 2.475] ]

        Returns
        -------
        media : A list with the mean of the distance 
                between the best x seen for each 
                evaluation and the argmin
    """
    r = []
    different_iteration = len(list_of_res)
    for res in list_of_res:
        r.append(list(convergence_res_x(res, min)))
    a = []
    media = []
    for i in range(len(list_of_res[0].func_vals)):
        for j in range(different_iteration):
            a.append(r[j][i])
        media.append(np.mean(a, dtype=np.float64))
        a = []
    return media


def my_key_fun(res):
    """
        Sort key for fun_min function
    """
    return res.fun


def fun_min(list_of_res):
    """
        Return the min of a list of BO
    """
    min_res = min(list_of_res, key=my_key_fun)
    return [min_res.fun, min_res.x]


def tabella(list_of_list_of_res):
    """
        Given a list of Bayesian_optimization results
        return a list with name, mean, median, 
        standard deviation and min result founded
        for each Bayesian_optimization result

        Parameters
        ----------
        list_of_list_of_res : A list of Bayesian_optimization 
                            results 

        Returns
        -------
        lista : A list with name, mean, median, 
                standard deviation and min result founded
                for each Bayesian_optimization result
    """
    lista = []
    different_iteration = len(list_of_list_of_res[0])
    for i in list_of_list_of_res:
        fun_media = []
        for it in range(different_iteration):
            fun_media.append(i[0][it].fun)

        lista.append([i[1], np.mean(fun_media, dtype=np.float64), median(i[0]), np.std(fun_media, dtype=np.float64),
                      fun_min(i[0])])
        # nome, media, mediana, std, [.fun min, .x min]
    return lista


def my_key_sort(list_with_name):
    """
        Sort key for top_5 funcion
    """
    return list_with_name[0]


def top_5(list_of_list_of_res):
    """
        Given a list of Bayesian_optimization results
        find out the best 5 result confronting the 
        best mean result

        Parameters
        ----------
        list_of_list_of_res : A list of Bayesian_optimization 
                            results 

        Returns
        -------
        list_medie : A list of each .pkl file's name 
                    just saved
                    -    list_of_list_of_res = [[res_BO_1,"name_1", 1], [res_BO_2,"name_2", 2],etc.]
    """
    list_medie = []
    for i in list_of_list_of_res:
        list_medie.append([total_mean(i[0]), i[1], i[2]])
    list_medie.sort(key=my_key_sort)
    list_medie = list_medie[:5]
    return list_medie


def plot_bayesian_optimization_old(list_of_res, name_plot="plot_BO.png",
                                   log_scale=False, path=None, conv_min=True):
    """
        Save a plot of the result of a Bayesian_optimization 
        considering mean and standard deviation.

        Parameters
        ----------
        list_of_res : A Bayesian_optimization result

        name_plot : The name of the file you want to 
                    give to the plot

        log_scale : y log scale if True

        path : path where the plot file is saved

        conv_min : If True the convergence is for the min,
                    If False is for the max

    """
    if conv_min:
        media = total_mean(list_of_res)
    else:
        media = total_mean_max(list_of_res)
    array = [i for i in range(len(media))]
    plt.plot(array, media, color='blue', label="res")

    lista_early_stop = len_func_vals(list_of_res)
    max_list = max(lista_early_stop)
    flag = True
    for i in lista_early_stop:
        if i != max_list:
            if flag:
                plt.axvline(x=(i - 1), color='red', label="early stop")
                flag = False  # In this way it doesn't generate too many label
            else:
                plt.axvline(x=(i - 1), color='red')

    if log_scale:
        plt.yscale('log')

    # x_int = range(0, array[-1]+1)
    # plt.xticks(x_int)
    if conv_min:
        plt.ylabel('min f(x) after n calls')
    else:
        plt.ylabel('max f(x) after n calls')
    plt.xlabel('Number of calls n')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True)
    if path is None:
        plt.savefig(name_plot)  # save in the current working directory
    else:
        if path[-1] != '/':
            path = path + "/"
        current_dir = os.getcwd()  # current working directory
        os.chdir(path)  # change directory
        plt.savefig(name_plot)
        os.chdir(current_dir)  # reset directory to original

    plt.close()


def dump_BO(res, stringa='result', path=None):
    """
        Dump (save) the Bayesian_optimization result

        Parameters
        ----------
        res : A result of a Bayesian_optimization
                    run
        stringa : Name of the log file saved in .pkl 
                format after the run of the function

        Returns
        -------
        dump_name : The .pkl file's name 
                    just saved 
    """
    dump_name = None
    if path is None:
        name_file = stringa + '.pkl'
        dump(res, name_file)
        dump_name = name_file  # save in the current working directory
    else:
        current_dir = os.getcwd()  # current working directory
        os.chdir(path)  # change directory
        if path[-1] != '/':
            path = path + "/"

        name_file = stringa + '.pkl'
        dump(res, name_file)

        dump_name = path + name_file

        os.chdir(current_dir)  # reset directory to original

    return dump_name


def load_BO(dump_name):
    """
        Load a pkl files, it should have the 
        name returned from dump_BO to work 
        properly, as it follows:
        -   dump_name = dump_BO( res_gp_rosenbrock )
        -   res_loaded = load_BO( dump_name )

        Parameters
        ----------
        dump_name : A name of a .pkl files

        Returns
        -------
        res_loaded : A Bayesian_optimization result
    """

    res_loaded = load(dump_name)
    return res_loaded


def plot_boxplot(matrix, name_plot="plot_model.png", path=None):
    """
        Save a boxplot of the data.
        Works only when optimization_runs is 1.

        Parameters
        ----------
        matrix: list of list of list of numbers
                or a 3D matrix
        
        name_plot : The name of the file you want to 
                    give to the plot

        path : path where the plot file is saved
    """

    matrix = matrix.transpose()
    # print(matrix)

    # fig7, ax7 = plt.subplots()
    plt.subplots()

    # ax7.set_title('Model runs')

    plt.xlabel('number of calls')
    plt.grid(True)

    # ax7.boxplot(matrix)
    plt.boxplot(matrix)

    if path is None:
        plt.savefig(name_plot)  # save in the current working directory
    else:
        if path[-1] != '/':
            path = path + "/"
        current_dir = os.getcwd()  # current working directory
        os.chdir(path)  # change directory
        plt.savefig(name_plot)
        os.chdir(current_dir)  # reset directory to original

    plt.close()


def plot_bayesian_optimization(res, name_plot="plot_BO.png",
                               log_scale=False, path=None, conv_min=True):
    """
        Save a convergence plot of the result of a 
        Bayesian_optimization.

        Parameters
        ----------
        res : A Bayesian_optimization result

        name_plot : The name of the file you want to 
                    give to the plot

        log_scale : y log scale if True

        path : path where the plot file is saved

        conv_min : If True the convergence is for the min,
                    If False is for the max

    """
    # print("RES plot->", res)
    if conv_min:
        media = convergence_res(res)
    else:
        media = convergence_res_max(res)

    array = [i for i in range(len(media))]
    plt.plot(array, media, color='blue', label="res")

    if (log_scale):
        plt.yscale('log')

    # x_int = range(0, array[-1]+1)
    # plt.xticks(x_int)
    if conv_min:
        plt.ylabel('min f(x) after n calls')
    else:
        plt.ylabel('max f(x) after n calls')
    plt.xlabel('Number of calls n')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True)
    if path is None:
        plt.savefig(name_plot)  # save in the current working directory
    else:
        if path[-1] != '/':
            path = path + "/"
        current_dir = os.getcwd()  # current working directory
        os.chdir(path)  # change directory
        plt.savefig(name_plot)
        os.chdir(current_dir)  # reset directory to original

    plt.close()
