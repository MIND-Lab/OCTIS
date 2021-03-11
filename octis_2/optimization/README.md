Optimize a model
----------------

To optimize a model you need to select a dataset, a metric and the search space of the hyperparameters to optimize.

```python
from optimization.optimizer import Optimizer

search_space = {
"alpha": Real(low=0.001, high=5.0),
"eta": Real(low=0.001, high=5.0)
}

# Initialize an optimizer object and start the optimization.
optimizer = Optimizer(model, dataset, metric, search_space)
result = optimizer.optimize()
```

 
The result will provide best-seen value of the metric with the corresponding hyperparameter configuration. To visualize this information you can set the `save` paramater of `Bayesian_optimization()` to True and a `.csv` file will save all the data. You can also set the `plot_best_seen` and the `plot_model` to True to save, respectively, the regression of the optimization and the box plot of the different model_runs for each point. 

Bayesian Optimization
---------------------
Bayesian_optimization is the core function.

```python
Bayesian_optimization(f, bounds, minimizer, number_of_call, model_runs, kernel, acq_func,
                      base_estimator_forest, random_state, noise_level, alpha, kappa,
                      x0, y0, time_x0, n_random_starts, save, save_step, save_name,save_path, 
                      early_stop, early_step plot_best_seen, plot_prefix_name, 
                      log_scale_plot, verbose, n_points, xi, n_jobs, model_queue_size)
                     
```
You can performe:
- Random Search setting the minimizer to `skopt.dummy_minimize`.
- Gaussian Process setting the minimizer to `skopt.gp_minimize`.
- Random Forest setting the minimizer to `skopt.forest_minimize` and the base_estimator_forest to `RF`.
- Extra Tree setting the minimizer to `skopt.forest_minimize` and the base_estimator_forest to `ET`.

To know more you could see the [[Code]](https://github.com/MIND-Lab/topic-modeling-evaluation-framework/blob/29f2ce28f7b03fa65f12933680eed61d2d6ee09b/optimization/optimizer.py#L231-L386) or some [[Examples]](https://github.com/MIND-Lab/topic-modeling-evaluation-framework/tree/master/examples). 

Nomenclature
------------
`f` : Function to minimize. Should take a single list of parameters and return the objective value.
      For example: 
```python
#Rosenbrock function of 2 variables
#min{f(x)} = 0
#argmin{f(x)} -> (1, 1) = (a, a**2)
def rosenbrock(x, a = 1, b = 100, noise_level = 0):
    return (a-x[0])**2 + b* ((x[1]-x[0]**2))**2 + ( noise_level ** 2 ) * np.random.randn()
```

`number_of_call` : [integer] Number of calls to f, therefore number of points evaluated.

`model_runs` : [integer] Number of different evaluation of the function in the same point and with the same hyperparameters. Usefull with a lot of noise, to reduce it.
`save` : [boolean] Save policy. It will save all the data in a `.csv` file and the optimization in a `.pkl` file.

`early_stop` : [boolean] Early stop policy. It will stop an optimization_run if it doesn't improve for early_step evaluations.

`plot_best_seen` : [boolean] Plot the convergence of the Bayesian optimization process, showing mean and standard deviation of the different optimization runs. If save is True the plot is update every save_step evaluations.

`plot_model` : [boolean] Plot the mean and standard deviation of the different model runs. If save is True the plot is update every save_step evaluations.

`save_models` : [boolean] If True it will save the model in a `.npz` file named as `<n_calls>_<model_runs>.npz`. Where `<n_calls>` is the actual n_calls and `<model_runs>` is the actual model_runs.
            
To know more you could see the [[Code]](https://github.com/MIND-Lab/topic-modeling-evaluation-framework/tree/29f2ce28f7b03fa65f12933680eed61d2d6ee09b/optimization) or [[Skopt]](https://scikit-optimize.github.io/stable/index.html). 

Examples
------------
You can find examples [[Here]](https://github.com/MIND-Lab/topic-modeling-evaluation-framework/tree/master/examples).
If you want to see a simple example of the optimization you can see the file [optimization_demo_BO.py](https://github.com/MIND-Lab/topic-modeling-evaluation-framework/blob/master/examples/optimization_demo_BO.py)




