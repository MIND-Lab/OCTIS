Local dashboard
================

The local dashboard is a user-friendly graphical interface for creating, monitoring, and viewing experiments.
Following the implementation standards of datasets, models, and metrics the dashboard will automatically update and allow you to use your custom implementations.

To run rhe dashboard you need to clone the repo.
While in the project directory run the following command:

.. code-block:: bash

    python OCTIS/dashboard/server.py --port [port number] --dashboardState [path to dashboard state file]

The port parameter is optional and the selected port number will be used to host the dashboard server, the default port is 5000.
The dashboardState parameter is optional and the selected json file will be used to save the informations used to launch and find the experiments, the default dashboardState path is the current directory.

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
In this tab, you can choose which batch (or set of batches) to visualize.

A plot of each experiment that contains the best-seen evaluation at each iteration is visualized in a grid.
Alternatively, you can visualize a box plot at each iteration to understand if a given hyper-parameter configuration is noisy (high variance) or not. 

You can interact with the single experiment graphic or choose to have a look at the single experiment by clicking on ``Click here to inspect the results``.

It is possible to decide in which order to show the experiments and apply some filters to have a more intuitive visualization of the experiments.


Visualize a custom experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the ``VISUALIZE EXPERIMENTS`` tab, after clicking on the ``Click here to inspect the results`` button, you will be redirected to the single experiment tab.
In this tab, you can visualize all the information and statistics related to the experiment, including the best hyper-parameter configuration and the best value of the optimized metric. You can also have an outline of the statistics of the tracked metrics. 

It is also possible to have a look at a word cloud obtained from the most relevant words of a given topic, scaled by their probability; the topic distribution on each document (and a preview of the document), and the weight of each word of the vocabulary for each topic. 


Manage the experiment queue
^^^^^^^^^^^^^^^^^^^^^^^^^^^
To manage the experiment queue click on the ``MANAGE EXPERIMENTS`` tab.
In this tab, you can pause or resume the execution of an experiment.
You can also change the order of the experiments to perform or delete the ones you are no longer interested in.


Frequently used terms
---------------------

Batch
^^^^^
A batch of experiments is a set of related experiments that can be recognized using a keyword referred to as batch ``name``.

Model runs
^^^^^^^^^^
In the optimization context of the framework, since the performance estimated by the evaluation metrics can be affected by noise, the objective function is computed as the median of a given number of ``model runs`` (i.e., topic models run with the same hyperparameter configuration)