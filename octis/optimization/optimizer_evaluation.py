import json
import numpy as np
import pandas as pd
from octis.optimization.optimizer_tool import check_instance, save_search_space, convert_type


class OptimizerEvaluation:

    def __init__(self, optimizer, BO_results):
        """
        Create an object with all the information about Bayesian Optimization

        :param optimizer: list of setting of the BO experiment
        :param BO_results: object of Scikit-optimize where the results of BO  are saved
        """
        search_space = optimizer.search_space
        optimization_type = optimizer.optimization_type

        # Creation of model metric-parameters saved in the json file
        metric_parameters = optimizer.metric.get_params()
        dict_metric_parameters = dict()

        for mp in metric_parameters:
            if check_instance(getattr(optimizer.metric, mp)):
                dict_metric_parameters.update({mp: getattr(optimizer.metric, mp)})

        # Creation of model hyper-parameters saved in the json file
        model_parameters = optimizer.model.hyperparameters
        dict_model_parameters = dict()

        for key in list(model_parameters.keys()):
            if check_instance(model_parameters[key]):
                dict_model_parameters.update({key: model_parameters[key]})

        # Creation of extra metric-parameters saved in the json file
        dict_extra_metric_parameters = dict()

        for em, em_name in zip(optimizer.extra_metrics, optimizer.extra_metric_names):
            metric_parameters = em.get_params()
            dict_extra_metric_parameters.update({em_name: dict()})
            for mp in metric_parameters:
                if check_instance(getattr(em, mp)):
                    dict_extra_metric_parameters[em_name].update({mp: getattr(em, mp)})

        # Info about optimization
        self.info = dict()
        dataset_info = optimizer.dataset.get_metadata()["info"]
        if dataset_info is not None:
            self.info.update({"dataset_name": dataset_info["name"]})
        else:
            self.info.update({"dataset_name": "dataset_name"})
        self.info.update({"dataset_path": optimizer.dataset.dataset_path})
        self.info.update({"is_cached": optimizer.dataset.is_cached})
        self.info.update({"kernel": str(optimizer.kernel)})
        self.info.update({"acq_func": optimizer.acq_func})
        self.info.update({"surrogate_model": optimizer.surrogate_model})
        self.info.update({"optimization_type": "Maximize" if optimization_type == "Maximize" else "Minimize"})
        self.info.update({"model_runs": optimizer.model_runs})
        self.info.update({"save_models": optimizer.save_models})
        self.info.update({"save_step": optimizer.save_step})
        self.info.update({"save_name": optimizer.save_name})
        self.info.update({"save_path": optimizer.save_path})
        self.info.update({"early_stop": optimizer.early_stop})
        self.info.update({"early_step": optimizer.early_step})
        self.info.update({"plot_model": optimizer.plot_model})
        self.info.update({"plot_best_seen": optimizer.plot_best_seen})
        self.info.update({"plot_name": optimizer.plot_name})
        self.info.update({"log_scale_plot": optimizer.log_scale_plot})
        self.info.update({"search_space": save_search_space(optimizer.search_space)})
        self.info.update({"model_name": optimizer.model.__class__.__name__})
        self.info.update({"model_attributes": dict_model_parameters})
        self.info.update({"use_partitioning": optimizer.model.use_partitions})
        self.info.update({"metric_name": optimizer.name_optimized_metric})
        self.info.update({"extra_metric_names": [name for name in optimizer.extra_metric_names]})
        self.info.update({"metric_attributes": dict_metric_parameters})
        self.info.update({"extra_metric_attributes": dict_extra_metric_parameters})
        self.info.update({"current_call": optimizer.current_call})
        self.info.update({"number_of_call": optimizer.number_of_call})
        self.info.update({"random_state": optimizer.random_state})
        self.info.update({"x0": optimizer.x0})
        self.info.update({"y0": optimizer.y0})
        self.info.update({"n_random_starts": optimizer.n_random_starts})
        self.info.update({"initial_point_generator": optimizer.initial_point_generator})
        self.info.update({"topk": optimizer.topk})
        self.info.update({"time_eval": optimizer.time_eval})
        self.info.update({"dict_model_runs": optimizer.dict_model_runs})

        # Reverse the sign of minimization if the problem is a maximization
        if optimization_type == "Maximize":
            self.func_vals = [-val for val in BO_results.func_vals]
            self.y_best = BO_results.fun
        else:
            self.func_vals = [val for val in BO_results.func_vals]
            self.y_best = BO_results.fun

        self.x_iters = dict()
        name_hyperparameters = sorted(list(search_space.keys()))

        # dictionary of x_iters
        lenList = len(BO_results.x_iters)
        for i, name in enumerate(name_hyperparameters):
            self.x_iters.update(
                {name: [convert_type(BO_results.x_iters[j][i]) for j in range(lenList)]})

        self.info.update({"f_val": self.func_vals})
        self.info.update({"x_iters": self.x_iters})

        self.metric = optimizer.metric
        self.extra_metrics = optimizer.extra_metrics

    def save(self, name_file):
        """
        Save results for Bayesian Optimization in a json file

        :param name_file: name of the file
        :type name_file: str
        """
        self.name_json = name_file
        with open(name_file, 'w') as fp:
            json.dump(self.info, fp)

    def save_to_csv(self, name_file):
        """
        Save results for Bayesian Optimization to a csv file

        :param name_file: name of the file
        :type name_file: str
        """
        n_row = len(self.func_vals)
        n_extra_metrics = len(self.extra_metrics)

        # creation of the Dataframe
        df = pd.DataFrame()
        df['dataset'] = [self.info["dataset_name"]] * n_row
        df['surrogate model'] = [self.info["surrogate_model"]] * n_row
        df['acquisition function'] = [self.info["acq_func"]] * n_row
        df['num_iteration'] = [i for i in range(n_row)]
        df['time'] = [self.info['time_eval'][i] for i in range(n_row)]
        df['Median(model_runs)'] = [np.median(
            self.info['dict_model_runs'][self.info['metric_name']]['iteration_' + str(i)]) for i in range(n_row)]
        df['Mean(model_runs)'] = [np.mean(
            self.info['dict_model_runs'][self.info['metric_name']]['iteration_' + str(i)]) for i in range(n_row)]
        df['Standard_Deviation(model_runs)'] = [np.std(
            self.info['dict_model_runs'][self.metric.__class__.__name__]['iteration_' + str(i)]) for i in range(n_row)]

        for hyperparameter in list(self.x_iters.keys()):
            df[hyperparameter] = self.x_iters[hyperparameter]

        for metric, i in zip(self.extra_metrics, range(n_extra_metrics)):
            try:
                df[metric.info()["name"] + '(not optimized)'] = [np.median(
                    self.dict_model_runs[metric.__class__.__name__]['iteration_' + str(i)]) for i in range(n_row)]
            except:
                df[metric.__class__.__name__ + '(not optimized)'] = [np.median(
                    self.dict_model_runs[metric.__class__.__name__]['iteration_' + str(i)]) for i in range(n_row)]

        if not name_file.endswith(".csv"):
            name_file = name_file + ".csv"

        # save the Dataframe to a csv
        df.to_csv(name_file, index=False, na_rep='Unknown')

    def load(self, name):
        """
        Load the results for Bayesian Optimization

        :param name: name of the json file
        :type name: str
        :return: dictionary of the results load from the json file
        :rtype: dict
        """
        with open(name, 'rb') as file:
            result = json.load(file)

        return result
