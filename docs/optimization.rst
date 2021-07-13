================================
Hyper-parameter optimization
================================

The core of OCTIS framework consists of an efficient and user-friendly way to select the best hyper-parameters for a Topic Model
using Bayesian Optimization.

To inizialize an optimization, inizialize the Optimizer class:

.. code-block:: bash

    from octis.optimization.optimizer import Optimizer
    optimizer = Optimizer()

Choose the dataset you want to analyze.

.. code-block:: bash

    from octis.dataset.dataset import Dataset
    dataset = Dataset()
    dataset.load("octis/preprocessed_datasets/M10")

Choose a Topic-Model.

.. code-block:: bash

     from octis.models.LDA import LDA
     model = LDA()
     model.hyperparameters.update({"num_topics": 25})

Choose the metric function to optimize.

.. code-block:: bash

    from octis.evaluation_metrics.coherence_metrics import Coherence
    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi = Coherence(metric_parameters)

Create the search space for the optimization.

.. code-block:: python

    from skopt.space.space import Real
    search_space = {
        "alpha": Real(low=0.001, high=5.0),
        "eta": Real(low=0.001, high=5.0)
    }

Finally, launch the optimization.

.. code-block:: python

    optimization_result=optimizer.optimize(model,
                       dataset,
                       npmi,
                       search_space,
                       number_of_call=10,
                       n_random_starts=3,
                       model_runs=3,
                       save_name="result",
                       surrogate_model="RF",
                       acq_func="LCB"
                        )

where:

* number_of_call: int, default: 5. Number of function evaluations.
* n_random_starts: int, default: 1. Number of random points used to inizialize the BO
* model_runs: int: default: 3. Number of model runs.
* save_name: str, default "results". Name of the json file where all the results are saved
* surrogate_model: str, default: "RF". Probabilistic surrogate model used to build to prior on the objective function. Can be either:

    * "RF" for Random Forest regression
    * "GP" for Gaussian Process regression
    * "ET" for Extra-tree Regression

* acq_function: str, default: "EI".  function to optimize the surrogate model. Can be either:

    * "LCB" for lower confidence bound
    * "EI" for expected improvment
    * "PI" for probability of improvment

The results of the optimization are saved in the json file, by default. However, you can save the results of the optimization also in a user-friendly csv file

.. code-block:: python

    optimization_result.save_to_csv("results.csv")

Resume the optimization
-------------------------

Optimization runs, for some reason, can be interrupted. With the help of the ``resume_optimization``  you can restart the optimization run from the last saved iteration.

.. code-block:: python

    optimizer = Optimizer()
    optimizer.resume_optimization(json_path)

where ``json_path`` is  the path of json file of the previous results.

Continue the optimization
-------------------------

Suppose that, after an optimization process, you want to perform three extra-evaluations.
You can do this using the method ``resume_optimization``.

.. code-block:: python

    optimizer = Optimizer()
    optimizer.resume_optimization(json_path, extra_evaluations=3)

where ``extra_evaluations`` (int, default 0) is the number of extra-evaluations to perform.

Inspect an extra-metric
-------------------------

Suppose that, during the optimization process, you want to inspect the value of another metric.
For example, suppose that you want to check the value of

.. code-block:: python

    metric_parameters = {
        'texts': dataset.get_corpus(),
        'topk': 10,
        'measure': 'c_npmi'
    }
    npmi2 = Coherence(metric_parameters)

You can add this as a parameter.

.. code-block:: python

    optimization_result=optimizer.optimize(model,
                       dataset,
                       npmi,
                       search_space,
                       number_of_call=10,
                       n_random_starts=3,
                       extra_metrics=[npmi2]
                        )

where ``extra_metrics`` (list, default None) is the list of extra metrics to inspect.

Early stopping
---------------

Suppose that you want to terminate the optimization process if there is no improvement after a certain number of iterations. You can apply an early stopping criterium during the optimization.


.. code-block:: python

    optimization_result=optimizer.optimize(model,
                       dataset,
                       npmi,
                       search_space,
                       number_of_call=10,
                       n_random_starts=3,
                       early_stop=True,
                       early_step=5,
                        )

where ``early_step`` (int, default 5) is the number of function evaluations after that the optimization process is stopped.
