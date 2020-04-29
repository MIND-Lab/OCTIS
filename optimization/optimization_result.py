class Evaluation():
    """
    Representation of a single optimization iteration result
    """
    hyperparameters = {}
    function_values = {}

    def __init__(self, hyperparameters_names, hyperparameters_values,
                 function_values):
        """
        Initialize result

        Parameters
        ----------
        hyperparameters_names : list of hyperparameters names
        hyperparameters_values : list of hyperparameters values
        function_values : dictionary of computed metrics values
                          key: metric name
                          value: metric value
        """

        hyperparameters = self._make_params_dict(
            hyperparameters_names,
            hyperparameters_values
        )

        self.hyperparameters = hyperparameters
        self.function_values = function_values

    def _make_params_dict(self, hyperparameters_names,
                          hyperparameters_values):
        """
        Create dictionary of hyperparameters
        from the list of name and the list of values

        Parameters
        ----------
        hyperparameters_names : list of hyperparameters names
        hyperparameters_values : list of hyperparameters values

        Returns
        -------
        params : dictionary of hyperparameters
                 key: hyperparameter name
                 value: hyperparameter value
        """
        params = {}
        for i in range(len(hyperparameters_names)):
            params[hyperparameters_names[i]] = hyperparameters_values[i]
        return params


class Best_evaluation(Evaluation):
    """
    Representation of the optimized values and each iteration
    """
    iterations = []

    def __init__(self, params_names, params_values, iters, optimized_metric):
        """
        Initialize result

        Parameters
        ----------
        params_names : list of hyperparameters names
        params_values : list of hyperparameters values of the best result
        iters : list of params_values and funcction_values of each iteration
        """
        iterations = []
        self.optimized_metric = optimized_metric
        function_values = {}
        for i in range(len(iters)):

            # If best hyperparameters are founded save their function values
            if iters[i][0] == params_values:
                function_values = iters[i][1]

            # Save iteration informations
            iterations.append(
                Evaluation(params_names,
                           iters[i][0],
                           iters[i][1]
                           )
            )
        super().__init__(params_names, params_values, function_values)

        self.iterations = iterations
