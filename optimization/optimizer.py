import numpy as np
from skopt import forest_minimize
from skopt.space.space import Real, Integer
from skopt.utils import dimensions_aslist
from optimization.optimization_result import Best_evaluation


class Optimizer():
    """
    Optimizer optimize hyperparameters to build topic models
    """

    # Values of hyperparameters and metrics for each iteration
    _iterations = []

    topk = 10  # if False the topk words will not be computed
    topic_word_matrix = True  # if False the matrix will not be computed
    topic_document_matrix = True  # if False the matrix will not be computed

    def __init__(self, model, dataset, metric, search_space, optimization_parameters={}):
        """
        Inititalize the optimizer for the model

        Parameters
        ----------
        model : model with hyperparameters to optimize
        metric : initialized metric to use for optimization
        search_space : a dictionary of hyperparameters to optimize
                       (each parameter is defined as a skopt space)
                       with the name of the hyperparameter given as key
        optimization_parameters : parameters of the random forest search
        """
        self.model = model
        self.dataset = dataset
        self.metric = metric
        self.search_space = search_space
        self.optimization_parameters = optimization_parameters

    def _objective_function(self, hyperparameters):
        """
        objective function to optimize

        Parameters
        ----------
        hyperparameters : dictionary of hyperparameters
                          key: name of the parameter
                          value: skopt search space dimension

        Returns
        -------
        result : score of the metric to maximize
        """

        # Retrieve parameters labels
        params = {}
        for i in range(len(self.hyperparameters)):
            params[self.hyperparameters[i]] = hyperparameters[i]

        # Prepare model
        model_output = self.model.train_model(
            self.dataset,
            params,
            self.topk,
            self.topic_word_matrix,
            self.topic_document_matrix)

        # Get metric score
        result = self.metric.score(model_output)

        # Update metrics values for extra metrics
        metrics_values = {self.metric.__class__.__name__: result}
        iteration = [hyperparameters, metrics_values]
        for extra_metric in self.extra_metrics:

            extra_metric_name = extra_metric.__class__.__name__
            if extra_metric_name not in metrics_values:
                name = extra_metric_name
            else:
                i = 2
                name = extra_metric_name + " 2"
                while name in metrics_values:
                    i += 1
                    name = extra_metric_name + " "+str(i)

            metrics_values[name] = extra_metric.score(model_output)

        # Save iteration data
        self._iterations.append(iteration)

        if self.optimization_type == 'Maximize':
            result = - result

        return result

    def optimize(self):
        """
        Optimize the hyperparameters of the model using random forest

        Parameters
        ----------


        Returns
        -------
        result : Best_evaluation object
        """
        self._iterations = []

        # Save parameters labels to use
        self.hyperparameters = list(sorted(self.search_space.keys()))
        params_space_list = dimensions_aslist(self.search_space)

        # Initialize default random forest parameters
        rf_parameters = {
            'n_calls': 100,
            'n_random_starts': 10,
            'acq_func': "LCB",
            'random_state': None,
            'verbose': False,
            'n_points': 10000,
            'base_estimator': 'RF',
            'kappa': 1.96,
            'x0': None,
            'y0': None,
            'xi': 1.96,
            'n_jobs': 1,
            'model_queue_size': None,
            'callback': None,
            'optimization_type': 'Maximize',
            'extra_metrics': []
        }

        # Customize random forest parameters
        rf_parameters.update(self.optimization_parameters)
        self.extra_metrics = rf_parameters["extra_metrics"]

        self.optimization_type = rf_parameters['optimization_type']

        optimize_result = forest_minimize(
            func=self._objective_function,
            dimensions=params_space_list,
            n_calls=rf_parameters["n_calls"],
            n_random_starts=rf_parameters["n_random_starts"],
            acq_func=rf_parameters["acq_func"],
            random_state=rf_parameters["random_state"],
            verbose=rf_parameters["verbose"],
            n_points=rf_parameters["n_points"],
            base_estimator=rf_parameters["base_estimator"],
            x0=rf_parameters["x0"],
            y0=rf_parameters["y0"],
            xi=rf_parameters["xi"],
            kappa=rf_parameters["kappa"],
            callback=rf_parameters["callback"],
            n_jobs=rf_parameters["n_jobs"],
            model_queue_size=rf_parameters["model_queue_size"]
        )

        if self.optimization_type == 'Maximize':
            optimize_result.fun = - optimize_result.fun
            optimize_result.func_vals = - optimize_result.func_vals

        # Create Best_evaluation object from optimization results
        result = Best_evaluation(self.hyperparameters,
                                 optimize_result,
                                 self._iterations,
                                 self.metric.__class__.__name__)

        return result
