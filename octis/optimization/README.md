Optimize a model
----------------

To optimize a model you need to select a dataset, a metric and the search space of the hyperparameters to optimize.

```python
from optimization.optimizer import Optimizer
from skopt.space.space import Real

search_space = {
"alpha": Real(low=0.001, high=5.0),
"eta": Real(low=0.001, high=5.0)
}
# Initialize an optimizer object and start the optimization.
optimizer = Optimizer()
result = optimizer.optimize(model, dataset, metric, search_space)
```

Plotting functions can be used to visualize the optimization process. To visualize the results you can set `plot_best_seen` and the `plot_model` to True to save, respectively, the convergence plot and the box plot of the different model_runs, for each iteration.

Bayesian Optimization
---------------------
Bayesian_optimization is the core function.

```python

optimize(self, model, dataset, metric, search_space, extra_metrics=None,
                 number_of_call=5, n_random_starts=1,
                 initial_point_generator="lhs",
                 optimization_type='Maximize', model_runs=5, surrogate_model="RF",
                 kernel=1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5),
                 acq_func="LCB", random_state=False, x0=None, y0=None,
                 save_models=True, save_step=1, save_name="result", save_path="results/", early_stop=False,
                 early_step=5,
                 plot_best_seen=False, plot_model=False, plot_name="B0_plot", log_scale_plot=False, topk=10)
```
To know more you could see the [[Code]](https://octis.readthedocs.io/en/latest/modules.html?highlight=optimizer#octis.optimization.optimizer.Optimizer)

The results of the optimization are saved in the json file, by default. However, you can save the results of the optimization also in a user-friendly csv file.

```python

optimization_result.save_to_csv("results.csv")

```

To know more you could see the [[Code]](https://github.com/MIND-Lab/OCTIS/blob/master/docs/optimization.rst)
