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
                optimization_runs,
                model_runs,
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
                plot_optimization,
                plot_model,
                plot_name,
                log_scale_plot,
                verbose,
                model_queue_size,
                checkpoint_saver,
                dataset_name,
                hyperparameters_name,
                metric_name,
                num_topic,
                maximize):
    res = []
    minimizer_stringa = "gp_minimize"

    if plot_optimization:
        if plot_name.endswith(".png") :
            name = plot_name
        else:
            name = plot_name + ".png"

    gpr = GaussianProcessRegressor(kernel=kernel, 
                                    alpha=alpha,
                                    normalize_y=True, 
                                    noise="gaussian",
                                    n_restarts_optimizer=0,
                                    random_state = random_state)

    if(not save and not early_stop ):
        for i in range( optimization_runs ):
            
            opt = skopt_optimizer(bounds, 
                            base_estimator=gpr, 
                            acq_func=acq_func,
                            n_random_starts = n_random_starts,
                            n_initial_points= n_random_starts,
                            acq_optimizer="sampling", 
                            random_state=random_state,
                            model_queue_size=model_queue_size )

            if( x0[i] is not None and y0[i] is not None):
                opt.tell(x0[i], y0[i], fit=True)
            res.append( opt.run(f, number_of_call) )

        if plot_optimization:
            tool.plot_bayesian_optimization( res, name, log_scale_plot, path=save_path )

    elif ( ( save_step >= number_of_call and save) and  ( early_step >= number_of_call or not early_stop )  ):
        for i in range( optimization_runs ):

            opt = skopt_optimizer(bounds, 
                            base_estimator=gpr, 
                            acq_func=acq_func,
                            n_random_starts = n_random_starts,
                            n_initial_points= n_random_starts,
                            acq_optimizer="sampling", 
                            random_state=random_state,
                            model_queue_size=model_queue_size )

            if( x0[i] is not None and y0[i] is not None):
                opt.tell(x0[i], y0[i], fit=True)

            res_t = opt.run(f, number_of_call)
            res.append( res_t )

        checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save

        if plot_optimization:
            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

    elif save and not early_stop:

        time_eval = []

        time_t = []
        for i in range( optimization_runs ):

            start_time = time.time()

            opt = skopt_optimizer(bounds, 
                            base_estimator=gpr, 
                            acq_func=acq_func,
                            n_random_starts = n_random_starts,
                            n_initial_points= n_random_starts,
                            acq_optimizer="sampling", 
                            random_state=random_state,
                            model_queue_size=model_queue_size )

            if( x0[i] is not None and y0[i] is not None):
                opt.tell(x0[i], y0[i], fit=True)

            res_t = opt.run(f, save_step)
            res.append( res_t )

            end_time = time.time()
            total_time = end_time - start_time
            time_t.append(total_time)

        time_eval.append(time_t)

        save_csv(name_csv = save_name + ".csv",
                dataset_name = dataset_name , 
                hyperparameters_name = hyperparameters_name,
                metric_name = metric_name,
                num_topic = num_topic, 
                Surrogate = minimizer_stringa,
                Acquisition = acq_func,
                Time = time_eval, 
                res = res,
                Maximize = maximize,
                time_x0 = time_x0 )


        checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
        number_of_call_r = number_of_call - save_step

        if plot_optimization:
            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )
        

        time_t = []
        while ( number_of_call_r > 0 ) :
            if( number_of_call_r >= save_step ):
                partial_res = tool.load_BO( checkpoint_saver ) #restore

                for i in range( optimization_runs ):
                    x0_restored = partial_res[i].x_iters
                    y0_restored = list(partial_res[i].func_vals)

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

                    res_t = opt.run(f, save_step)
                    res[i] = res_t

                    end_time = time.time()
                    total_time = end_time - start_time
                    time_t.append(total_time)

                time_eval.append(time_t)

                save_csv(name_csv = save_name + ".csv",
                        dataset_name = dataset_name , 
                        hyperparameters_name = hyperparameters_name,
                        metric_name = metric_name,
                        num_topic = num_topic, 
                        Surrogate = minimizer_stringa,
                        Acquisition = acq_func,
                        Time = time_eval, 
                        res = res,
                        Maximize = maximize,
                        time_x0 = time_x0 )

                checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                number_of_call_r = number_of_call_r - save_step

                if plot_optimization:
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

            else:
                partial_res = tool.load_BO( checkpoint_saver ) #restore
                for i in range( optimization_runs ):
                    x0_restored = partial_res[i].x_iters
                    y0_restored = list(partial_res[i].func_vals)

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

                    res_t = opt.run(f, number_of_call_r)
                    res[i] = res_t

                    end_time = time.time()
                    total_time = end_time - start_time
                    time_t.append(total_time)

                time_eval.append(time_t)

                save_csv(name_csv = save_name + ".csv",
                        dataset_name = dataset_name , 
                        hyperparameters_name = hyperparameters_name,
                        metric_name = metric_name,
                        num_topic = num_topic, 
                        Surrogate = minimizer_stringa,
                        Acquisition = acq_func,
                        Time = time_eval, 
                        res = res,
                        Maximize = maximize,
                        time_x0 = time_x0 )

                checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                number_of_call_r = number_of_call_r - save_step

                if plot_optimization:
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

    elif (not save and early_stop):

        early_stop_flag = [False] * optimization_runs
        
        for i in range( optimization_runs ):
            if( early_stop_flag[i] == False ):

                opt = skopt_optimizer(bounds, 
                                base_estimator=gpr, 
                                acq_func=acq_func,
                                n_random_starts = n_random_starts,
                                n_initial_points= n_random_starts,
                                acq_optimizer="sampling", 
                                random_state=random_state,
                                model_queue_size=model_queue_size )

                if( x0[i] is not None and y0[i] is not None):
                    opt.tell(x0[i], y0[i], fit=True)

                res_t = opt.run(f, early_step)
                if tool.early_condition(res_t, early_step, n_random_starts):
                    early_stop_flag[i] = True

                res.append( res_t )

        

        checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
        number_of_call_r = number_of_call - early_step

        if plot_optimization:
            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )
        

        while ( number_of_call_r > 0 ) :
            if( number_of_call_r >= early_step ):
                partial_res = tool.load_BO( checkpoint_saver ) #restore

                for i in range( optimization_runs ):
                    if( early_stop_flag[i] == False ):
                        x0_restored = partial_res[i].x_iters
                        y0_restored = list(partial_res[i].func_vals)

                        opt = skopt_optimizer(bounds, 
                                base_estimator=gpr, 
                                acq_func=acq_func,
                                n_random_starts = 0,
                                n_initial_points= 0,
                                acq_optimizer="sampling", 
                                random_state=random_state,
                                model_queue_size=model_queue_size )

                        opt.tell(x0_restored, y0_restored, fit=True)

                        res_t = opt.run(f, early_step)
                        if tool.early_condition(res_t, early_step, n_random_starts):
                            early_stop_flag[i] = True

                        res[i] = res_t

                checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                number_of_call_r = number_of_call_r - early_step

                if plot_optimization:
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

            else:
                partial_res = tool.load_BO( checkpoint_saver ) #restore
                for i in range( optimization_runs ):
                    if( early_stop_flag[i] == False ):
                        x0_restored = partial_res[i].x_iters
                        y0_restored = list(partial_res[i].func_vals)

                        opt = skopt_optimizer(bounds, 
                                base_estimator=gpr, 
                                acq_func=acq_func,
                                n_random_starts = 0,
                                n_initial_points= 0,
                                acq_optimizer="sampling", 
                                random_state=random_state,
                                model_queue_size=model_queue_size )

                        opt.tell(x0_restored, y0_restored, fit=True)

                        res_t = opt.run(f, number_of_call_r)
                        if tool.early_condition(res_t, early_step, n_random_starts):
                            early_stop_flag[i] = True
                            
                        res[i] = res_t

                checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                number_of_call_r = number_of_call_r - early_step

                if plot_optimization:
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

    elif ( save and early_stop):

        early_stop_flag = [False] * optimization_runs
        
        for i in range( optimization_runs ):
            if( early_stop_flag[i] == False ):

                opt = skopt_optimizer(bounds, 
                                base_estimator=gpr, 
                                acq_func=acq_func,
                                n_random_starts = n_random_starts,
                                n_initial_points= n_random_starts,
                                acq_optimizer="sampling", 
                                random_state=random_state,
                                model_queue_size=model_queue_size )

                if( x0[i] is not None and y0[i] is not None):
                    opt.tell(x0[i], y0[i], fit=True)

                res_t = opt.run(f, save_step)
                if tool.early_condition(res_t, early_step, n_random_starts):
                    early_stop_flag[i] = True

                res.append( res_t )

        

        checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
        number_of_call_r = number_of_call - save_step

        if plot_optimization:
            tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )
        

        while ( number_of_call_r > 0 ) :
            if( number_of_call_r >= save_step ):
                partial_res = tool.load_BO( checkpoint_saver ) #restore

                for i in range( optimization_runs ):
                    if( early_stop_flag[i] == False ):
                        x0_restored = partial_res[i].x_iters
                        y0_restored = list(partial_res[i].func_vals)

                        opt = skopt_optimizer(bounds, 
                                base_estimator=gpr, 
                                acq_func=acq_func,
                                n_random_starts = 0,
                                n_initial_points= 0,
                                acq_optimizer="sampling", 
                                random_state=random_state,
                                model_queue_size=model_queue_size )

                        opt.tell(x0_restored, y0_restored, fit=True)

                        res_t = opt.run(f, save_step)
                        if tool.early_condition(res_t, early_step, n_random_starts):
                            early_stop_flag[i] = True

                        res[i] = res_t

                checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                number_of_call_r = number_of_call_r - save_step

                if plot_optimization:
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

            else:
                partial_res = tool.load_BO( checkpoint_saver ) #restore
                for i in range( optimization_runs ):
                    if( early_stop_flag[i] == False ):
                        x0_restored = partial_res[i].x_iters
                        y0_restored = list(partial_res[i].func_vals)

                        opt = skopt_optimizer(bounds, base_estimator=gpr,
                                                acq_func=acq_func, n_random_starts=0,
                                                n_initial_points=0, acq_optimizer="sampling",
                                                random_state=random_state,
                                                model_queue_size=model_queue_size )

                        opt.tell(x0_restored, y0_restored, fit=True)

                        res_t = opt.run(f, number_of_call_r)
                        if tool.early_condition(res_t, early_step, n_random_starts):
                            early_stop_flag[i] = True
                            
                        res[i] = res_t

                checkpoint_saver = tool.dump_BO( res, save_name, save_path ) #save
                number_of_call_r = number_of_call_r - save_step

                if plot_optimization:
                    tool.plot_bayesian_optimization( res, name, log_scale_plot, path = save_path )

    else:
        print("Not implemented \n")

    return res