from skopt import dummy_minimize
import optimization.optimizer_tool as tool
from optimization.csv_creator import save_csv
from optimization.stopper import MyCustomEarlyStopper

from skopt import load
from skopt.callbacks import CheckpointSaver
import time


# Dummy Minimize
def random_minimizer(f,
                     bounds,
                     number_of_call,
                     random_state,
                     x0,
                     y0,
                     time_x0,
                     n_random_starts,
                     save,
                     save_step,
                     save_name,
                     save_path,
                     early_stop,
                     early_step,
                     plot_best_seen,
                     plot_prefix_name,
                     log_scale_plot,
                     verbose,
                     model_queue_size,
                     dataset_name,
                     hyperparameters_name,
                     metric_name,
                     maximize,
                     acq_func="None"):
    
    res = None
    minimizer_stringa = "random_minimize"

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
        res = dummy_minimize(f,
                            bounds,
                            n_calls=number_of_call,
                            x0=x0,
                            y0=y0,
                            random_state=random_state,
                            verbose=verbose,
                            model_queue_size=model_queue_size)

    elif ((save_step >= number_of_call and save) and (early_step >= number_of_call or not early_stop)):
        save_name_t = save_name + ".pkl"
        checkpoint_saver = CheckpointSaver(save_name_t)  # save

        res = dummy_minimize(f,
                            bounds,
                            n_calls=number_of_call,
                            x0=x0,
                            y0=y0,
                            random_state=random_state,
                            callback=[checkpoint_saver],
                            verbose=verbose,
                            model_queue_size=model_queue_size)

    elif (save and not early_stop):

        time_eval = []

        #time_t = []
        save_name_t = save_name + ".pkl"
        checkpoint_saver = CheckpointSaver(save_name_t)  # save

        start_time = time.time()

        res = dummy_minimize(f,
                            bounds,
                            n_calls=save_step,
                            x0=x0,
                            y0=y0,
                            random_state=random_state,
                            callback=[checkpoint_saver],
                            verbose=verbose,
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

        number_of_call_r = number_of_call - save_step

        #time_t = []
        while (number_of_call_r > 0):
            if (number_of_call_r >= save_step):
                
                save_name_t = save_name + ".pkl"
                partial_res = load(save_name_t)  # restore
                x0_restored = partial_res.x_iters
                y0_restored = partial_res.func_vals
                save_name_t = "./" + save_name + ".pkl"
                checkpoint_saver_t = CheckpointSaver(save_name_t)  # save

                start_time = time.time()

                res = dummy_minimize(f,
                                        bounds,
                                        n_calls=save_step,
                                        x0=x0_restored,
                                        y0=y0_restored,
                                        callback=[checkpoint_saver],
                                        random_state=random_state,
                                        verbose=verbose,
                                        model_queue_size=model_queue_size)

                checkpoint_saver = checkpoint_saver_t

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

            else:
                save_name_t = save_name + ".pkl"
                partial_res = load(save_name_t)  # restore
                x0_restored = partial_res.x_iters
                y0_restored = partial_res.func_vals

                start_time = time.time()

                res = dummy_minimize(f, bounds, n_calls=number_of_call_r,
                                        x0=x0_restored, y0=y0_restored,
                                        callback=[checkpoint_saver],
                                        random_state=random_state,
                                        verbose=verbose,
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
                        time_x0=time_x0)

                if plot_best_seen:
                    tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)
                number_of_call_r = number_of_call_r - save_step

    elif not save and early_stop:
        
        res = dummy_minimize(f,
                            bounds,
                            n_calls=number_of_call,
                            x0=x0,
                            y0=y0,
                            callback=[MyCustomEarlyStopper(
                                n_stop=early_step,
                                n_random_starts=n_random_starts)
                            ],
                            random_state=random_state,
                            verbose=verbose,
                            model_queue_size=model_queue_size)

    elif save and early_stop:

        time_eval = []
        save_name_t = save_name + ".pkl"
        checkpoint_saver = CheckpointSaver(save_name_t)  # save

        start_time = time.time()

        res = dummy_minimize(f,
                            bounds,
                            n_calls=save_step,
                            x0=x0,
                            y0=y0,
                            random_state=random_state,
                            callback=[checkpoint_saver,
                                    MyCustomEarlyStopper(
                                        n_stop=early_step,
                                        n_random_starts=n_random_starts)],
                            verbose=verbose,
                            model_queue_size=model_queue_size)

        end_time = time.time()
        total_time = end_time - start_time
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

        number_of_call_r = number_of_call - save_step

        while (number_of_call_r > 0):

            if (number_of_call_r >= save_step):
                save_name_t = save_name + ".pkl"
                partial_res = load(save_name_t)  # restore
                x0_restored = partial_res.x_iters
                y0_restored = partial_res.func_vals
                save_name_t = "./" + save_name + ".pkl"
                checkpoint_saver_t = CheckpointSaver(save_name_t)  # save

                start_time = time.time()

                res = dummy_minimize(f,
                                        bounds,
                                        n_calls=save_step,
                                        x0=x0_restored,
                                        y0=y0_restored,
                                        callback=[checkpoint_saver,
                                                    MyCustomEarlyStopper(
                                                        n_stop=early_step,
                                                        n_random_starts=n_random_starts)],
                                        random_state=random_state,
                                        verbose=verbose,
                                        model_queue_size=model_queue_size)

                end_time = time.time()
                total_time = end_time - start_time
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

                checkpoint_saver = checkpoint_saver_t

                number_of_call_r = number_of_call_r - save_step

                if plot_best_seen:
                    tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

            else:
                save_name_t = save_name + ".pkl"
                partial_res = load(save_name_t)  # restore
                x0_restored = partial_res.x_iters
                y0_restored = partial_res.func_vals

                start_time = time.time()

                res = dummy_minimize(f, bounds, n_calls=number_of_call_r,
                                        x0=x0_restored, y0=y0_restored,
                                        callback=[checkpoint_saver,
                                                    MyCustomEarlyStopper(
                                                        n_stop=early_step,
                                                        n_random_starts=n_random_starts)],
                                        random_state=random_state,
                                        verbose=verbose,
                                        model_queue_size=model_queue_size)

                end_time = time.time()
                total_time = end_time - start_time
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
                        time_x0=time_x0)

                number_of_call_r = number_of_call_r - save_step

                if plot_best_seen:
                    tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

    else:
        print("Not implemented \n")

    if plot_best_seen:
            tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

    return res
