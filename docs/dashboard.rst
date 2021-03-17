Local dashboard
================

The local dashboard is a user friendly graphical interface for creating, monitoring and viewing experiments.
Following the implementation standards of datasets, models and metrics the dashboard will automatically update and allow you to use your own custom implementations.

To run rhe dashboard, while in the project directory run the following command:

.. code-block:: bash

    python OCTIS/dashboard/server.py

The browser will open and you will be redirected to the dashboard.
In the dashboard you can:

* Create new experiments organized in batch
* Visualize and compare all the experiments
* Visualize a custom experiment
* Manage the experiment queue

Using the Dashboard
-------------------

When the dashboard opens, the home will be automatically loaded on your browser.

Create new experiments
^^^^^^^^^^^^^^^^^^^^^^
To create a new experiment click on the ``CREATE EXPERIMENTS`` tab.
In this tab have to choose:

* The folder in which you want to save the experiment results
* The name of the experiment
* The name of the batch of experiments in which the experiment is contained
* The dataset
* The model to optimize
* Hyperparameters of the model to optimize
* Search space of the hyperparameters to optimize
* The metric to optimize
* Parameters of the metric
* Metrics to track [optional]
* Parameters of the metrics to track [optional]
* Optimization parameters

After that you can click on ``Start Experiment`` and the experiment will be added to the Queue.

Visualize and compare all the experiments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To visualize the experiments click on the ``VISUALIZE EXPERIMENTS`` tab.
In this tab you can choose which bach (or set of batchs) to visualize.

The experiments will be shown in a grid, where it will be possible to decide in which order to show them and apply some filters.

You can interact with the single experiment graphic or choose to have a look to the single experiment by clickig on ``Click here to inspect the results``.


Visualize a custom experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the ``VISUALIZE EXPERIMENTS`` tab, after clicking on the ``Click here to inspect the results`` button, you will be redirected to the single experiment tab.
In this tab you can look to the collected data of each iteration and model run of each experiment.

Manage the experiment queue
^^^^^^^^^^^^^^^^^^^^^^^^^^^
To manage the experiment queue click on the ``MANAGE EXPERIMENTS`` tab.
In this tab you can pause or resume the execution of an experiment.
You can also change the order of the experiments to perform, or delete the ones you are no longer interested in.