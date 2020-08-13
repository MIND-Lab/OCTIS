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
                     optimization_runs,
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
                     plot_model,
                     plot_prefix_name,
                     log_scale_plot,
                     verbose,
                     model_queue_size,
                     checkpoint_saver,
                     dataset_name,
                     hyperparameters_name,
                     metric_name,
                     maximize,
                     acq_func="None"):
    res = []
    minimizer_stringa = "random_minimize"

    if plot_best_seen:
        if plot_prefix_name.endswith(".png") :
            plot_best_seen_name = plot_prefix_name[:-4] + "_best_seen.png"
        else:
            plot_best_seen_name = plot_prefix_name + "_best_seen.png"

    if save_path is not None:
        save_name = save_path + save_name

    if not save and not early_stop:
        for i in range(optimization_runs):
            res.append(dummy_minimize(f,
                                      bounds,
                                      n_calls=number_of_call,
                                      x0=x0[i],
                                      y0=y0[i],
                                      random_state=random_state,
                                      verbose=verbose,
                                      model_queue_size=model_queue_size))

        if plot_best_seen:
            tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

    elif ((save_step >= number_of_call and save) and (early_step >= number_of_call or not early_stop)):
        for i in range(optimization_runs):
            save_name_t = save_name + str(i) + ".pkl"
            checkpoint_saver[i] = CheckpointSaver(save_name_t)  # save

            res.append(dummy_minimize(f,
                                      bounds,
                                      n_calls=number_of_call,
                                      x0=x0[i],
                                      y0=y0[i],
                                      random_state=random_state,
                                      callback=[checkpoint_saver[i]],
                                      verbose=verbose,
                                      model_queue_size=model_queue_size))

        if plot_best_seen:
            tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

    elif (save and not early_stop):

        time_eval = []

        time_t = []
        for i in range(optimization_runs):
            save_name_t = save_name + str(i) + ".pkl"
            checkpoint_saver[i] = CheckpointSaver(save_name_t)  # save

            start_time = time.time()

            res.append(dummy_minimize(f,
                                      bounds,
                                      n_calls=save_step,
                                      x0=x0[i],
                                      y0=y0[i],
                                      random_state=random_state,
                                      callback=[checkpoint_saver[i]],
                                      verbose=verbose,
                                      model_queue_size=model_queue_size))

            end_time = time.time()
            total_time = end_time - start_time
            time_t.append(total_time)

        time_eval.append(time_t)

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

        number_of_call_r = number_of_call - save_step

        time_t = []
        while (number_of_call_r > 0):
            if (number_of_call_r >= save_step):
                for i in range(optimization_runs):
                    save_name_t = save_name + str(i) + ".pkl"
                    partial_res = load(save_name_t)  # restore
                    x0_restored = partial_res.x_iters
                    y0_restored = partial_res.func_vals
                    save_name_t = "./" + save_name + str(i) + ".pkl"
                    checkpoint_saver_t = CheckpointSaver(save_name_t)  # save

                    start_time = time.time()

                    res[i] = dummy_minimize(f,
                                            bounds,
                                            n_calls=save_step,
                                            x0=x0_restored,
                                            y0=y0_restored,
                                            callback=[checkpoint_saver[i]],
                                            random_state=random_state,
                                            verbose=verbose,
                                            model_queue_size=model_queue_size)

                    checkpoint_saver[i] = checkpoint_saver_t

                    end_time = time.time()
                    total_time = end_time - start_time
                    time_t.append(total_time)

                time_eval.append(time_t)

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

            else:
                for i in range(optimization_runs):
                    save_name_t = save_name + str(i) + ".pkl"
                    partial_res = load(save_name_t)  # restore
                    x0_restored = partial_res.x_iters
                    y0_restored = partial_res.func_vals

                    start_time = time.time()

                    res[i] = dummy_minimize(f, bounds, n_calls=number_of_call_r,
                                            x0=x0_restored, y0=y0_restored,
                                            callback=[checkpoint_saver[i]],
                                            random_state=random_state,
                                            verbose=verbose,
                                            model_queue_size=model_queue_size)

                    end_time = time.time()
                    total_time = end_time - start_time
                    time_t.append(total_time)

                time_eval.append(time_t)

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
        for i in range(optimization_runs):
            res_temp = dummy_minimize(f,
                                      bounds,
                                      n_calls=number_of_call,
                                      x0=x0[i],
                                      y0=y0[i],
                                      callback=[MyCustomEarlyStopper(
                                          n_stop=early_step,
                                          n_random_starts=n_random_starts)
                                      ],
                                      random_state=random_state,
                                      verbose=verbose,
                                      model_queue_size=model_queue_size)
            res.append(res_temp)

        if plot_best_seen:
            tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

    elif save and early_stop:

        for i in range(optimization_runs):
            save_name_t = save_name + str(i) + ".pkl"
            checkpoint_saver[i] = CheckpointSaver(save_name_t)  # save

            res_temp = dummy_minimize(f,
                                      bounds,
                                      n_calls=save_step,
                                      x0=x0[i],
                                      y0=y0[i],
                                      random_state=random_state,
                                      callback=[checkpoint_saver[i],
                                                MyCustomEarlyStopper(
                                                    n_stop=early_step,
                                                    n_random_starts=n_random_starts)],
                                      verbose=verbose,
                                      model_queue_size=model_queue_size)

            res.append(res_temp)

        if plot_best_seen:
            tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

        number_of_call_r = number_of_call - save_step

        while (number_of_call_r > 0):

            if (number_of_call_r >= save_step):
                for i in range(optimization_runs):
                    save_name_t = save_name + str(i) + ".pkl"
                    partial_res = load(save_name_t)  # restore
                    x0_restored = partial_res.x_iters
                    y0_restored = partial_res.func_vals
                    save_name_t = "./" + save_name + str(i) + ".pkl"
                    checkpoint_saver_t = CheckpointSaver(save_name_t)  # save

                    res[i] = dummy_minimize(f,
                                            bounds,
                                            n_calls=save_step,
                                            x0=x0_restored,
                                            y0=y0_restored,
                                            callback=[checkpoint_saver[i],
                                                      MyCustomEarlyStopper(
                                                          n_stop=early_step,
                                                          n_random_starts=n_random_starts)],
                                            random_state=random_state,
                                            verbose=verbose,
                                            model_queue_size=model_queue_size)

                    checkpoint_saver[i] = checkpoint_saver_t

                number_of_call_r = number_of_call_r - save_step

                if plot_best_seen:
                    tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)


            else:
                for i in range(optimization_runs):
                    save_name_t = save_name + str(i) + ".pkl"
                    partial_res = load(save_name_t)  # restore
                    x0_restored = partial_res.x_iters
                    y0_restored = partial_res.func_vals

                    res[i] = dummy_minimize(f, bounds, n_calls=number_of_call_r,
                                            x0=x0_restored, y0=y0_restored,
                                            callback=[checkpoint_saver[i],
                                                      MyCustomEarlyStopper(
                                                          n_stop=early_step,
                                                          n_random_starts=n_random_starts)],
                                            random_state=random_state,
                                            verbose=verbose,
                                            model_queue_size=model_queue_size)

                number_of_call_r = number_of_call_r - save_step

                if plot_best_seen:
                    tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

        if plot_best_seen:
            tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

    else:
        print("Not implemented \n")

    return res
