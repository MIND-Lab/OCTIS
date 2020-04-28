import numpy as np
from skopt import forest_minimize
from skopt.space.space import Real, Integer
from skopt.utils import dimensions_aslist


class Optimizer():
    """
    Optimizer optimize hyperparameters to build topic models
    """

    # Parameters of the metric
    metric_parameters = {}

    topk = 10  # if False the topk words will not be computed
    topic_word_matrix = True  # if False the matrix will not be computed
    topic_document_matrix = True  # if False the matrix will not be computed

    def __init__(self, model, metric, search_space,
        metric_parameters={}, optimization_parameters={}):
        """
        Inititalize the optimizer for the model

        Parameters
        ----------
        model : model with hyperparameters to optimize
        metric : metric to use for optimization
        search_space : a dictionary of hyperparameters to optimize
                       (each parameter is defined as a skopt space)
                       with the name of the hyperparameter given as key
        metric_parameters : dictionary with extra parameters of the metric
        optimization_parameters : parameters of the random forest search
        """
        self.model = model
        self.metric = metric
        self.metric_parameters.update(metric_parameters)
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

        self.model.set_hyperparameters(params)

        self.model.train_model()

        model_output = self.model.get_output(
            self.topk,
            self.topic_word_matrix,
            self.topic_document_matrix)

        metric = self.metric(model_output, self.metric_parameters)
        result = - metric.score()
        return result

    def optimize(self):
        """
        Optimize the hyperparameters of the model using random forest

        Parameters
        ----------
        

        Returns
        -------
        result : list [params, optimize_result]
                 params: dictionary with optimized hyperparameters
                 optimize_result: optimization result as an object
                 (skopt OptimizeResult format)
        """

        # Save parameters labels to use
        self.hyperparameters = list(sorted(self.search_space.keys()))
        params_space_list = dimensions_aslist(self.search_space)

        # Initialize default random forest parameters
        rf_parameters = {
            'n_calls': 100,
            'n_random_starts': 10,
            'acq_func': "EI",
            'random_state': None,
            'verbose': False,
            'n_points': 10000,
            'base_estimator': 'ET',
            'kappa': 1.96,
            'x0': None,
            'y0': None,
            'xi': 1.96,
            'n_jobs': 1,
            'model_queue_size': None,
            'callback': None
        }

        # Customize random forest parameters
        rf_parameters.update(self.optimization_parameters)

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
        
        optimize_result.fun = - optimize_result.fun
        optimize_result.func_vals = - optimize_result.func_vals

        # Associate parameters label with their best value after optimization
        params = {}
        for i in range(len(self.hyperparameters)):
            params[self.hyperparameters[i]] = optimize_result.x[i]

        result = [params, optimize_result]
        return result
