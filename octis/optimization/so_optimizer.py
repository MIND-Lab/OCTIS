import numpy as np
from dragonfly import maximise_function, minimise_function
from dragonfly import load_config, load_config_file
from argparse import Namespace
from octis.models.model import save_model_output
from pathlib import Path
import os
import pickle

class SOOptimizer:

    def __init__(self, model, dataset, search_space=None, config_file=None, model_runs=2, number_of_calls=10, metric=None,
                 progress_file=None, maximize=True, acq_func='ucb', save_results=True, save_path="results/"):

        self.model = model
        self.dataset = dataset
        self.model_runs = model_runs
        self.metric = metric
        self.progress_file = progress_file
        self.maximize = maximize
        self.number_of_calls = number_of_calls
        self.acq_func = acq_func

        self.save_models = save_results
        self.save_path = save_path
        if self.save_models:
            self.model_path_models = self.save_path + "models/"
            Path(self.model_path_models).mkdir(parents=True, exist_ok=True)
        if search_space is not None:
            config_params = {'domain': search_space}
            self.config = load_config(config_params)
        elif config_file is not None:
            self.config = load_config_file(config_file)
        self.options = Namespace(
            build_new_model_every=1,  # update the model every 5 iterations
            report_results_every=4,  # report progress every 4 iterations
            acq=self.acq_func
            # report_model_on_each_build=True,  # report model when you build it.
        )
        self.current_call = 0

        if self.progress_file is not None:
            self.options.progress_load_from_and_save_to = progress_file
            self.options.progress_save_every = 2
            if os.path.exists(progress_file):
                prog = pickle.load(open(progress_file, 'rb'))
                if 'config' in prog.keys():
                    self.current_call = len(prog['config']) + 1

    #@staticmethod
    def objective(self, x):
        metrics_results = []
        if os.path.exists(self.progress_file):
            prog = pickle.load(open(self.progress_file, 'rb'))
            if 'raw_points' in prog:
                if self.config.domain.get_type() == 'euclidean':
                    ordered_x = [x[name] for name in self.config.domain_orderings.raw_name_ordering]
                else:
                    ordered_x = [x[name] for name in self.config.domain.raw_name_ordering]

                for i, p in enumerate(prog['raw_points']):
                    if tuple(ordered_x) == tuple(p):
                        return prog['true_vals'][i]

        for i in range(self.model_runs):
            model_output = self.model.train_model(hyperparameters=x)
            if self.save_models:
                name = str(self.current_call) + "_" + str(i)
                save_model_path = self.model_path_models + name
                save_model_output(model_output, save_model_path)

            metrics_results.append(self.metric.score(model_output))

        self.current_call += 1

        return np.median(metrics_results)

    def optimize(self):
        if self.maximize:
            pareto_values, pareto_points, history = maximise_function(
                self.objective, domain=self.config.domain, max_capital=self.number_of_calls,
                config=self.config, opt_method='bo', options=self.options)
        else:
            pareto_values, pareto_points, history = minimise_function(
                self.objective, domain=self.config.domain, max_capital=self.number_of_calls,
                config=self.config, opt_method='bo', options=self.options)
        print(pareto_values)
        print(pareto_points)
