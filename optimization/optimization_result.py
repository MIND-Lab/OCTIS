import matplotlib.pyplot as plt
import skopt.plots

import matplotlib.backends.backend_pdf
import matplotlib


class Evaluation():
    """
    Representation of a single optimization iteration result
    """
    hyperparameters = {}
    function_values = {}

    def __init__(self, hyperparameters_names, hyperparameters_values,
                 function_values):
        """
        Initialize class

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
    optimized_result = None
    optimized_metric = None

    def __init__(self, params_names, optimized_result, iters, optimized_metric):
        """
        Initialize class

        Parameters
        ----------
        params_names : list of hyperparameters names
        optimized_result : OptimizeResult object
        iters : list of params_values and funcction_values of each iteration
        """
        self.optimized_result = optimized_result
        iterations = []
        self.optimized_metric = optimized_metric
        function_values = {}
        for i in range(len(iters)):

            # If best hyperparameters are founded save their function values
            if iters[i][0] == optimized_result.x:
                function_values = iters[i][1]

            # Save iteration informations
            iterations.append(
                Evaluation(params_names,
                           iters[i][0],
                           iters[i][1]
                           )
            )
        super().__init__(params_names, optimized_result.x, function_values)

        self.iterations = iterations

    def plot(self, iterations, metric=None):
        """
        Plot values of each iterations for a single metric

        Parameters
        ----------
        iterations : iterations
        metric : metric name 
                 default: optimized metric
        """
        if self.optimized_result != None:
            if metric == None:
                metric = self.optimized_metric

            # Create list of points to plot
            x = list(range(len(iterations)))
            y = []
            for i in x:
                y.append(iterations[i].function_values[metric])
            plt.plot(x, y)

    def plot_all(self, metric=None, hyperparameter=None,
                 extra_info=False, path=None):
        """
        Plot all the informations of the optimization

        Parameters
        ----------
        metric : name of the metric to use to sort iterations
                 if metric and hyperparameter are None,
                 iterations will be plotted in chronological order
        hyperparameter : name of the hyperparameter to use to sort iterations
                 if metric and hyperparameter are None,
                 iterations will be plotted in chronological order
        extra_info : True if skopt extra info are nedeed
                     default: False
        path : path in wich a pdf with all data will be saved
               default: None, data will be showed instead
        """

        # Create pool of figures
        figures = {}

        # Sort iterations values by metric
        if metric != None:
            iterations = sorted(self.iterations,
                                key=lambda i: i.function_values[metric])
        elif hyperparameter != None:
            iterations = sorted(self.iterations,
                                key=lambda i: i.hyperparameters[
                                    hyperparameter])
        else:
            iterations = self.iterations

        # Plot all metrics evaluations in different graphs
        for metric in self.function_values.keys():
            figures[metric] = plt.figure()
            plt.figure(figures[metric].number)
            figures[metric].suptitle(metric)
            self.plot(iterations, metric)

        # Create hyperparameters plot and axes
        parameters, axs = plt.subplots(len(self.hyperparameters))
        figures["parameters"] = parameters

        # Set hyperparameter name for the hyperparameters axes
        i = 0
        for param in self.hyperparameters.keys():
            axs[i].set_title(param)
            i += 1

        plt.figure(figures["parameters"].number)
        x = range(len(iterations))
        hyperparameters_points = [[] for i in range(len(self.hyperparameters))]

        # Create list of points to plot for each hyperparameter
        for i in x:
            j = 0
            for param in iterations[i].hyperparameters.keys():
                hyperparameters_points[j].append(
                    iterations[i].hyperparameters[param])
                j += 1

        # Plot each hyperparameter graph in the same figure
        for i in range(len(hyperparameters_points)):
            axs[i].plot(
                x,
                hyperparameters_points[i]
            )

        # Handle extra info from skopt
        if extra_info:
            axes = skopt.plots.plot_objective(self.optimized_result)
            axes.flatten()[0].figure.suptitle("objective")
            figures["objective"] = axes.flatten()[0].figure

            axes2 = skopt.plots.plot_evaluations(self.optimized_result)
            axes2.flatten()[0].figure.suptitle("evaluations")
            figures["evaluations"] = axes2.flatten()[0].figure

        # Create a nice formatting to show data
        for _, figure in figures.items():
            ID = figure.number
            plt.figure(ID)
            plt.subplots_adjust(
                left=0.15,
                right=0.85,
                top=0.85,
                bottom=0.15,
                wspace=0.22,
                hspace=0.22
            )

        if path == None:
            plt.show()
        else:
            # Save pdf with all data
            pdf = matplotlib.backends.backend_pdf.PdfPages(path+".pdf")
            for _, figure in figures.items():
                ID = figure.number
                pdf.savefig(ID)
            pdf.close()
