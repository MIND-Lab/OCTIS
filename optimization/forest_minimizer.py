from skopt import forest_minimize
import optimization.optimizer_tool as tool
from optimization.csv_creator import save_csv
from optimization.stopper import MyCustomEarlyStopper

from skopt import load
from skopt.callbacks import CheckpointSaver
import time
import numpy as np


# Forest Minimize
def forest_minimizer(f, bounds, number_of_call,
                    acq_func, base_estimator_forest, random_state, kappa, x0, y0,
                    time_x0, n_random_starts, save, save_step, save_name, save_path,
                    early_stop, early_step, plot_best_seen, plot_model,
                    plot_prefix_name, log_scale_plot, verbose, n_points,
                    xi, n_jobs, model_queue_size, checkpoint_saver,
                    dataset_name, hyperparameters_name, metric_name, maximize):
    res = None
    minimizer_stringa = "forest_minimize"

    if x0 is None:
        lenx0 = 0
    else:
        lenx0 = len(x0)


    if plot_best_seen:
        if plot_prefix_name.endswith(".png") :
            plot_best_seen_name = plot_prefix_name[:-4] + "_best_seen.png"
        else:
            plot_best_seen_name = plot_prefix_name + "_best_seen.png"

    if save_path is not None:
        save_name = save_path + save_name

    if not save and not early_stop:
        res = forest_minimize(f, bounds, base_estimator=base_estimator_forest,
                            n_calls=number_of_call, acq_func=acq_func,
                            n_random_starts=n_random_starts, x0=x0, y0=y0,
                            random_state=random_state, verbose=verbose,
                            n_points=n_points, xi=xi, kappa=kappa,
                            n_jobs=n_jobs, model_queue_size=model_queue_size)

    elif (save_step >= number_of_call and save) and (early_step >= number_of_call or not early_stop):
        save_name_t = save_name + ".pkl"
        checkpoint_saver = CheckpointSaver(save_name_t)  # save

        res = forest_minimize(f, bounds, base_estimator=base_estimator_forest,
                            n_calls=number_of_call, acq_func=acq_func,
                            n_random_starts=n_random_starts,
                            x0=x0, y0=y0, random_state=random_state,
                            callback=[checkpoint_saver], verbose=verbose,
                            n_points=n_points, xi=xi, kappa=kappa, n_jobs=n_jobs,
                            model_queue_size=model_queue_size)

    elif save and not early_stop:
        time_eval = []
        time_t = []
            
        save_name_t = save_name + ".pkl"
        checkpoint_saver = CheckpointSaver(save_name_t)  # save
        if x0 is None:
            len_x0 = 0
        else:
            len_x0 = len(x0)

        flag = False
        if save_step >= n_random_starts + len_x0:
            n_calls_t = save_step
        else:
            n_calls_t = save_step + n_random_starts  # + len_x0
            flag = True

        start_time = time.time()

        res = forest_minimize(f,
                            bounds,
                            base_estimator=base_estimator_forest,
                            n_calls=n_calls_t,
                            acq_func=acq_func,
                            n_random_starts=n_random_starts,
                            x0=x0,
                            y0=y0,
                            random_state=random_state,
                            callback=[checkpoint_saver],
                            verbose=verbose,
                            n_points=n_points,
                            xi=xi,
                            kappa=kappa,
                            n_jobs=n_jobs,
                            model_queue_size=model_queue_size)

        end_time = time.time()
        total_time = end_time - start_time
        time_t.append(total_time)

        if plot_best_seen:
            tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

        number_of_call_r = number_of_call - save_step
        if flag:
            fract = save_step + n_random_starts
        else:
            fract = number_of_call - number_of_call_r

        time_t = [i / fract for i in time_t]

        for i in range(fract):
            time_eval.append(time_t[0])

        save_csv(name_csv=save_name + ".csv", dataset_name=dataset_name,
                hyperparameters_name=hyperparameters_name, metric_name=metric_name,
                Surrogate=minimizer_stringa,
                Acquisition=acq_func, Time=time_eval,
                res=res, Maximize=maximize, time_x0=time_x0,
                len_x0=lenx0)

        time_t = []
        while number_of_call_r > 0:
            if number_of_call_r >= save_step:
                save_name_t = save_name + ".pkl"
                partial_res = load(save_name_t)  # restore
                x0_restored = partial_res.x_iters
                y0_restored = partial_res.func_vals
                save_name_t = "./" + save_name + ".pkl"
                checkpoint_saver_t = CheckpointSaver(save_name_t)  # save

                start_time = time.time()

                res = forest_minimize(f, bounds, base_estimator=base_estimator_forest,
                                        n_calls=save_step, acq_func=acq_func,
                                        n_random_starts=0, x0=x0_restored, y0=y0_restored,
                                        random_state=random_state,
                                        callback=[checkpoint_saver], verbose=verbose,
                                        n_points=n_points, xi=xi, kappa=kappa,
                                        n_jobs=n_jobs, model_queue_size=model_queue_size)

                checkpoint_saver = checkpoint_saver_t

                end_time = time.time()
                total_time = end_time - start_time
                time_eval.append(total_time)

                save_csv(name_csv=save_name + ".csv", dataset_name=dataset_name,
                        hyperparameters_name=hyperparameters_name, metric_name=metric_name,
                        Surrogate=minimizer_stringa,
                        Acquisition=acq_func,
                        Time=time_eval,
                        res=res,
                        Maximize=maximize,
                        time_x0=time_x0,
                        len_x0=lenx0)

                if plot_best_seen:
                    tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

                number_of_call_r = number_of_call_r - save_step

            else:
                save_name_t = save_name + ".pkl"
                partial_res = load(save_name_t)  # restore
                x0_restored = partial_res.x_iters
                y0_restored = partial_res.func_vals

                start_time = time.time()

                res = forest_minimize(f,
                                    bounds,
                                    base_estimator=base_estimator_forest,
                                    n_calls=number_of_call_r,
                                    acq_func=acq_func,
                                    n_random_starts=0,
                                    x0=x0_restored,
                                    y0=y0_restored,
                                    random_state=random_state,
                                    callback=[checkpoint_saver],
                                    verbose=verbose,
                                    n_points=n_points,
                                    xi=xi,
                                    kappa=kappa,
                                    n_jobs=n_jobs,
                                    model_queue_size=model_queue_size)

                end_time = time.time()
                total_time = end_time - start_time
                #time_t.append(total_time)

                time_eval.append(total_time)

                save_csv(name_csv=save_name + ".csv",
                        dataset_name=dataset_name,
                        hyperparameters_name=hyperparameters_name,
                        metric_name=metric_name,
                        Surrogate=minimizer_stringa,
                        Acquisition=acq_func,
                        Time=time_eval,
                        res=res,
                        Maximize=maximize,
                        time_x0=time_x0,
                        len_x0=lenx0)

                if plot_best_seen:
                    tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)
                number_of_call_r = number_of_call_r - save_step

    elif not save and early_stop:
        res = forest_minimize(f,
                            bounds,
                            base_estimator=base_estimator_forest,
                            n_calls=number_of_call,
                            acq_func=acq_func,
                            n_random_starts=n_random_starts,
                            x0=x0,
                            y0=y0,
                            random_state=random_state,
                            callback=[MyCustomEarlyStopper(
                                n_stop=early_step,
                                n_random_starts=n_random_starts)],
                            verbose=verbose,
                            n_points=n_points,
                            xi=xi,
                            kappa=kappa,
                            n_jobs=n_jobs,
                            model_queue_size=model_queue_size)

            #res.append(res_temp)

    elif save and early_stop:
        save_name_t = save_name + ".pkl"
        checkpoint_saver = CheckpointSaver(save_name_t)  # save

        if x0 is None:
            len_x0 = 0
        else:
            len_x0 = len(x0)

        if save_step >= n_random_starts + len_x0:
            n_calls_t = save_step
        else:
            n_calls_t = save_step + n_random_starts
            flag = True

        time_eval = []
        time_t = []

        start_time = time.time()

        res = forest_minimize(f,
                            bounds,
                            base_estimator=base_estimator_forest,
                            n_calls=n_calls_t,
                            acq_func=acq_func,
                            n_random_starts=n_random_starts,
                            x0=x0,
                            y0=y0,
                            random_state=random_state,
                            callback=[checkpoint_saver,
                                        MyCustomEarlyStopper(
                                            n_stop=early_step,
                                            n_random_starts=n_random_starts)],
                            verbose=verbose,
                            n_points=n_points,
                            xi=xi,
                            kappa=kappa,
                            n_jobs=n_jobs,
                            model_queue_size=model_queue_size)

        end_time = time.time()
        total_time = end_time - start_time
        time_t.append(total_time)

        if plot_best_seen:
            tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

        number_of_call_r = number_of_call - save_step
        if flag:
            fract = save_step + n_random_starts
        else:
            fract = number_of_call - number_of_call_r


        time_t = [i / fract for i in time_t]

        for i in range(fract):
            time_eval.append(time_t[0])

        save_csv(name_csv=save_name + ".csv", dataset_name=dataset_name,
                hyperparameters_name=hyperparameters_name, metric_name=metric_name,
                Surrogate=minimizer_stringa,
                Acquisition=acq_func,
                Time=time_eval,
                res=res,
                Maximize=maximize,
                time_x0=time_x0,
                len_x0=lenx0)

        if plot_best_seen:
            tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

        number_of_call_r = number_of_call - save_step

        while number_of_call_r > 0:
            if number_of_call_r >= save_step:
                save_name_t = save_name + ".pkl"
                partial_res = load(save_name_t)  # restore
                x0_restored = partial_res.x_iters
                y0_restored = partial_res.func_vals
                save_name_t = "./" + save_name + ".pkl"
                checkpoint_saver_t = CheckpointSaver(save_name_t)  # save

                if (x0 is None):
                    len_x0 = 0
                else:
                    len_x0 = len(x0)

                if (save_step >= len_x0):
                    n_calls_t = save_step
                else:
                    n_calls_t = save_step + len_x0

                start_time = time.time()

                res = forest_minimize(f, bounds, base_estimator=base_estimator_forest,
                                    n_calls=n_calls_t, acq_func=acq_func,
                                    n_random_starts=0, x0=x0_restored,
                                    y0=y0_restored, random_state=random_state,
                                    callback=[checkpoint_saver,
                                            MyCustomEarlyStopper(
                                                n_stop=early_step,
                                                n_random_starts=n_random_starts)],
                                    verbose=verbose, n_points=n_points, xi=xi,
                                    kappa=kappa, n_jobs=n_jobs,
                                    model_queue_size=model_queue_size)

                end_time = time.time()
                total_time = end_time - start_time
                time_eval.append(total_time)

                save_csv(name_csv=save_name + ".csv", dataset_name=dataset_name,
                        hyperparameters_name=hyperparameters_name, metric_name=metric_name,
                        Surrogate=minimizer_stringa,
                        Acquisition=acq_func,
                        Time=time_eval,
                        res=res,
                        Maximize=maximize,
                        time_x0=time_x0,
                        len_x0=lenx0)

                checkpoint_saver = checkpoint_saver_t

                number_of_call_r = number_of_call_r - save_step

                if plot_best_seen:
                    tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

            else:
                save_name_t = save_name + ".pkl"
                partial_res = load(save_name_t)  # restore
                x0_restored = partial_res.x_iters
                y0_restored = partial_res.func_vals

                if x0 is None:
                    len_x0 = 0
                else:
                    len_x0 = len(x0)

                if save_step >= n_random_starts + len_x0:
                    n_calls_t = number_of_call_r
                else:
                    n_calls_t = number_of_call_r + len_x0
                    flag = True

                start_time = time.time()

                res = forest_minimize(f,
                                    bounds,
                                    base_estimator=base_estimator_forest,
                                    n_calls=n_calls_t,
                                    acq_func=acq_func,
                                    n_random_starts=0,
                                    x0=x0_restored,
                                    y0=y0_restored,
                                    random_state=random_state,
                                    callback=[checkpoint_saver,
                                            MyCustomEarlyStopper(
                                                n_stop=early_step,
                                                n_random_starts=n_random_starts)],
                                    verbose=verbose,
                                    n_points=n_points,
                                    xi=xi,
                                    kappa=kappa,
                                    model_queue_size=model_queue_size)

                end_time = time.time()
                total_time = end_time - start_time
                time_eval.append(total_time)

                save_csv(name_csv=save_name + ".csv", dataset_name=dataset_name,
                        hyperparameters_name=hyperparameters_name, metric_name=metric_name,
                        Surrogate=minimizer_stringa,
                        Acquisition=acq_func,
                        Time=time_eval,
                        res=res,
                        Maximize=maximize,
                        time_x0=time_x0,
                        len_x0=lenx0)

                number_of_call_r = number_of_call_r - save_step

                if plot_best_seen:
                    tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

    else:
        print("Not implemented \n")

    if plot_best_seen:
            tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

    return res
