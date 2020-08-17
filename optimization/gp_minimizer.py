from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt import Optimizer as skopt_optimizer

import optimization.optimizer_tool as tool
from optimization.csv_creator import save_csv
from optimization.stopper import MyCustomEarlyStopper

from skopt import load
from skopt.callbacks import CheckpointSaver
import time

#base_estimator='gp', n_initial_points=10,  acq_optimizer='auto', acq_func_kwargs=None, acq_optimizer_kwargs=None)
#optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None, noise=None)

#GP Minimize
def gp_minimizer(f,
                bounds,
                number_of_call,
                kernel,
                acq_func,
                random_state,
                noise_level, #attenzione
                alpha,
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
                maximize):
    res = []
    minimizer_stringa = "gp_minimize"

    if x0 is None:
        lenx0 = 0
    else:
        lenx0 = len(x0)


    if plot_best_seen:
        if plot_prefix_name.endswith(".png") :
            plot_best_seen_name = plot_prefix_name[:-4] + "_best_seen.png"
        else:
            plot_best_seen_name = plot_prefix_name + "_best_seen.png"

    gpr = GaussianProcessRegressor(kernel=kernel, 
                                    alpha=alpha,
                                    normalize_y=True, 
                                    noise="gaussian",
                                    n_restarts_optimizer=0,
                                    random_state = random_state)

    if (not save) and (not early_stop):
            
        opt = skopt_optimizer(bounds, 
                        base_estimator=gpr, 
                        acq_func=acq_func,
                        n_random_starts = n_random_starts,
                        n_initial_points= n_random_starts,
                        acq_optimizer="sampling", 
                        random_state=random_state,
                        model_queue_size=model_queue_size )

        if( x0 is not None and y0 is not None):
            opt.tell(x0, y0, fit=True)
        res = opt.run(f, number_of_call)

        if plot_best_seen:
            tool.plot_bayesian_optimization( res, plot_best_seen_name, log_scale_plot, path=save_path )

    elif ( ( save_step >= number_of_call and save) and  ( early_step >= number_of_call or not early_stop )  ):

        time_eval = []
        start_time = time.time()

        opt = skopt_optimizer(bounds, 
                        base_estimator=gpr, 
                        acq_func=acq_func,
                        n_random_starts = n_random_starts,
                        n_initial_points= n_random_starts,
                        acq_optimizer="sampling", 
                        random_state=random_state,
                        model_queue_size=model_queue_size )

        if( x0 is not None and y0 is not None):
            opt.tell(x0, y0, fit=True)

        res = opt.run(f, number_of_call)

        end_time = time.time()
        total_time = end_time - start_time
        time_mean = total_time / number_of_call
        time_eval = [time_mean]*number_of_call
        if save:
            save_csv(name_csv = save_path + save_name + ".csv",
                    dataset_name = dataset_name , 
                    hyperparameters_name = hyperparameters_name,
                    metric_name = metric_name,
                    Surrogate = minimizer_stringa,
                    Acquisition = acq_func,
                    Time = time_eval, 
                    res = res,
                    Maximize = maximize,
                    time_x0 = time_x0,
                    len_x0=lenx0)

        checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save

        if plot_best_seen:
            tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

    elif save and not early_stop:

        time_eval = []

        start_time = time.time()

        opt = skopt_optimizer(bounds, 
                        base_estimator=gpr, 
                        acq_func=acq_func,
                        n_random_starts = n_random_starts,
                        n_initial_points= n_random_starts,
                        acq_optimizer="sampling", 
                        random_state=random_state,
                        model_queue_size=model_queue_size )

        if( x0 is not None and y0 is not None):
            opt.tell(x0, y0, fit=True)

        res = opt.run(f, save_step)

        end_time = time.time()
        total_time = end_time - start_time
        time_eval.append(total_time)

        save_csv(name_csv = save_path + save_name + ".csv",
                dataset_name = dataset_name , 
                hyperparameters_name = hyperparameters_name,
                metric_name = metric_name,
                Surrogate = minimizer_stringa,
                Acquisition = acq_func,
                Time = time_eval, 
                res = res,
                Maximize = maximize,
                time_x0 = time_x0,
                len_x0=lenx0)


        checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
        number_of_call_r = number_of_call - save_step

        if plot_best_seen:
            tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)
        


        while ( number_of_call_r > 0 ) :
            if( number_of_call_r >= save_step ):
                partial_res = tool.load_BO( checkpoint_saver ) #restore

                x0_restored = partial_res.x_iters
                y0_restored = list(partial_res.func_vals)

                start_time = time.time()

                opt = skopt_optimizer(bounds, 
                        base_estimator=gpr, 
                        acq_func=acq_func,
                        n_random_starts = 0,
                        n_initial_points= 0,
                        acq_optimizer="sampling", 
                        random_state=random_state,
                        model_queue_size=model_queue_size )


                opt.tell(x0_restored, y0_restored, fit=True)

                res = opt.run(f, save_step)

                end_time = time.time()
                total_time = end_time - start_time
                time_eval.append(total_time)

                save_csv(name_csv = save_path + save_name + ".csv",
                        dataset_name = dataset_name , 
                        hyperparameters_name = hyperparameters_name,
                        metric_name = metric_name,
                        Surrogate = minimizer_stringa,
                        Acquisition = acq_func,
                        Time = time_eval, 
                        res = res,
                        Maximize = maximize,
                        time_x0 = time_x0,
                        len_x0=lenx0 )

                checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                number_of_call_r = number_of_call_r - save_step

                if plot_best_seen:
                    tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

            else:
                partial_res = tool.load_BO( checkpoint_saver ) #restore
                
                x0_restored = partial_res.x_iters
                y0_restored = list(partial_res.func_vals)

                start_time = time.time()

                opt = skopt_optimizer(bounds, 
                        base_estimator=gpr, 
                        acq_func=acq_func,
                        n_random_starts = 0,
                        n_initial_points= 0,
                        acq_optimizer="sampling", 
                        random_state=random_state,
                        model_queue_size=model_queue_size )

                opt.tell(x0_restored, y0_restored, fit=True)

                res = opt.run(f, number_of_call_r)

                end_time = time.time()
                total_time = end_time - start_time
                time_eval.append(total_time)

                save_csv(name_csv = save_path + save_name + ".csv",
                        dataset_name = dataset_name , 
                        hyperparameters_name = hyperparameters_name,
                        metric_name = metric_name,
                        Surrogate = minimizer_stringa,
                        Acquisition = acq_func,
                        Time = time_eval, 
                        res = res,
                        Maximize = maximize,
                        time_x0 = time_x0,
                        len_x0=lenx0 )

                checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                number_of_call_r = number_of_call_r - save_step

                if plot_best_seen:
                    tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

    elif (not save) and early_stop:

        early_stop_flag = False
        
        if( early_step < number_of_call ):
            step = early_step
            number_of_call_r = number_of_call - early_step
        else:
            step = number_of_call
            number_of_call_r = 0


        opt = skopt_optimizer(bounds, 
                        base_estimator=gpr, 
                        acq_func=acq_func,
                        n_random_starts = n_random_starts,
                        n_initial_points= n_random_starts,
                        acq_optimizer="sampling", 
                        random_state=random_state,
                        model_queue_size=model_queue_size )

        if( x0 is not None and y0 is not None):
            opt.tell(x0, y0, fit=True)

        res = opt.run(f, step)
        if tool.early_condition(res, early_step, n_random_starts):
            early_stop_flag = True

        checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save

        if plot_best_seen:
            tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)
        

        while ( number_of_call_r > 0 ) :
            if( number_of_call_r >= early_step ):
                partial_res = tool.load_BO( checkpoint_saver ) #restore

                if( early_stop_flag == False ):
                    x0_restored = partial_res.x_iters
                    y0_restored = list(partial_res.func_vals)

                    opt = skopt_optimizer(bounds, 
                            base_estimator=gpr, 
                            acq_func=acq_func,
                            n_random_starts = 0,
                            n_initial_points= 0,
                            acq_optimizer="sampling", 
                            random_state=random_state,
                            model_queue_size=model_queue_size )

                    opt.tell(x0_restored, y0_restored, fit=True)

                    res = opt.run(f, early_step)
                    if tool.early_condition(res, early_step, n_random_starts):
                        early_stop_flag = True

                checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                number_of_call_r = number_of_call_r - early_step

                if plot_best_seen:
                    tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

            else:
                partial_res = tool.load_BO( checkpoint_saver ) #restore

                if( early_stop_flag == False ):
                    x0_restored = partial_res.x_iters
                    y0_restored = list(partial_res.func_vals)

                    opt = skopt_optimizer(bounds, 
                            base_estimator=gpr, 
                            acq_func=acq_func,
                            n_random_starts = 0,
                            n_initial_points= 0,
                            acq_optimizer="sampling", 
                            random_state=random_state,
                            model_queue_size=model_queue_size )

                    opt.tell(x0_restored, y0_restored, fit=True)

                    res = opt.run(f, number_of_call_r)
                    if tool.early_condition(res, early_step, n_random_starts):
                        early_stop_flag = True

                checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                number_of_call_r = number_of_call_r - early_step

                if plot_best_seen:
                    tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

    elif ( save and early_stop):

        early_stop_flag = False
        time_eval = []
        

        if( early_stop_flag == False ):

            start_time = time.time()

            opt = skopt_optimizer(bounds, 
                            base_estimator=gpr, 
                            acq_func=acq_func,
                            n_random_starts = n_random_starts,
                            n_initial_points= n_random_starts,
                            acq_optimizer="sampling", 
                            random_state=random_state,
                            model_queue_size=model_queue_size )

            if( x0 is not None and y0 is not None):
                opt.tell(x0, y0, fit=True)

            res = opt.run(f, save_step)

            end_time = time.time()
            total_time = end_time - start_time
            time_eval.append(total_time)

            save_csv(name_csv = save_path + save_name + ".csv",
                    dataset_name = dataset_name , 
                    hyperparameters_name = hyperparameters_name,
                    metric_name = metric_name,
                    Surrogate = minimizer_stringa,
                    Acquisition = acq_func,
                    Time = time_eval, 
                    res = res,
                    Maximize = maximize,
                    time_x0 = time_x0,
                    len_x0=lenx0)

            if tool.early_condition(res, early_step, n_random_starts):
                early_stop_flag = True

        

        checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
        number_of_call_r = number_of_call - save_step

        if plot_best_seen:
            tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)
        

        while ( number_of_call_r > 0 ) :
            if( number_of_call_r >= save_step ):
                partial_res = tool.load_BO( checkpoint_saver ) #restore

                if( early_stop_flag == False ):
                    x0_restored = partial_res.x_iters
                    y0_restored = list(partial_res.func_vals)

                    start_time = time.time()

                    opt = skopt_optimizer(bounds, 
                            base_estimator=gpr, 
                            acq_func=acq_func,
                            n_random_starts = 0,
                            n_initial_points= 0,
                            acq_optimizer="sampling", 
                            random_state=random_state,
                            model_queue_size=model_queue_size )

                    opt.tell(x0_restored, y0_restored, fit=True)

                    res = opt.run(f, save_step)

                    end_time = time.time()
                    total_time = end_time - start_time
                    time_eval.append(total_time)

                    save_csv(name_csv = save_path + save_name + ".csv",
                            dataset_name = dataset_name , 
                            hyperparameters_name = hyperparameters_name,
                            metric_name = metric_name,
                            Surrogate = minimizer_stringa,
                            Acquisition = acq_func,
                            Time = time_eval, 
                            res = res,
                            Maximize = maximize,
                            time_x0 = time_x0,
                            len_x0=lenx0)

                    if tool.early_condition(res, early_step, n_random_starts):
                        early_stop_flag = True

                checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                number_of_call_r = number_of_call_r - save_step

                if plot_best_seen:
                    tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

            else:
                partial_res = tool.load_BO( checkpoint_saver ) #restore

                if( early_stop_flag == False ):
                    x0_restored = partial_res.x_iters
                    y0_restored = list(partial_res.func_vals)
                    start_time = time.time()

                    opt = skopt_optimizer(bounds, base_estimator=gpr,
                                            acq_func=acq_func, n_random_starts=0,
                                            n_initial_points=0, acq_optimizer="sampling",
                                            random_state=random_state,
                                            model_queue_size=model_queue_size )

                    opt.tell(x0_restored, y0_restored, fit=True)

                    res = opt.run(f, number_of_call_r)

                    end_time = time.time()
                    total_time = end_time - start_time
                    time_eval.append(total_time)

                    save_csv(name_csv = save_path + save_name + ".csv",
                            dataset_name = dataset_name , 
                            hyperparameters_name = hyperparameters_name,
                            metric_name = metric_name,
                            Surrogate = minimizer_stringa,
                            Acquisition = acq_func,
                            Time = time_eval, 
                            res = res,
                            Maximize = maximize,
                            time_x0 = time_x0,
                            len_x0=lenx0)

                    if tool.early_condition(res, early_step, n_random_starts):
                        early_stop_flag = True

                checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                number_of_call_r = number_of_call_r - save_step

                if plot_best_seen:
                    tool.plot_bayesian_optimization(res, plot_best_seen_name, log_scale_plot, path=save_path)

    else:
        print("Not implemented \n")

    return res