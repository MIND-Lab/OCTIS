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

 
The result will provide best-seen value of the metric with the corresponding hyperparameter configuration, and the hyperparameters and metric value for each iteration of the optimization. To visualize this information, you can use the plot and plot_all methods of the result.

Bayesian_optimization
----------------

Bayesian_optimization is the core function.
