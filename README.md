# Topic modeling evaluation framework


Evaluation framework that compare topic models' performance with respect to different metrics. The topic models are optimized using bayesian optimization techniques

Features
--------

* Acquire datasets from multiple sources
* Preprocess datasets
* Build topic models from a list of implemented  models, or create your own
* Evaluate models using a set of evaluation metrics
* Optimize models with respect to a metric

Acquire dataset
---------------

To acquire a dataset you can use one of the built-in sources.

.. code-block:: python

    import preprocessing.sources.newsgroup as source
    dataset = source.retrieve()

Or use your own.

.. code-block:: python

    import preprocessing.sources.custom_dataset as source
    dataset = source.retrieve("path\to\dataset")

A custom dataset must have a document for each line of the file.

Preprocess
----------

To preprocess a dataset Initialize a Pipeline_handler and use the preprocess method.

.. code-block:: python

    from preprocessing.pipeline_handler import Pipeline_handler
    pipeline_handler = Pipeline_handler(dataset)
    preprocessed = pipeline_handler.preprocess()
    preprocessed.save("dataset_folder") # Save the preprocessed dataset

For the customization of the preprocess pipeline see the optimization demo example example in the examples folder.

Build a model
-------------

To build a model first load a preprocessed dataset.

.. code-block:: python

    from dataset.dataset import Dataset
    dataset = Dataset()
    dataset.load("dataset_folder")

Set the hyperparameters.

.. code-block:: python
    
    hyperparameters = {}
    hyperparameters["num_topics"] = 20

And build your model

.. code-block:: python

    from models.LDA import LDA_Model
    model = LDA_Model(dataset, hyperparameters)  # Create model
    model.train_model()  # Train the model

Evaluate a model
----------------

To evaluate a model, choose ametric and follow two simple steps

Set the metric hyperparameters.

.. code-block:: python
    
    td_parameters ={
    'topk':10
    }

And compute the metric.

.. code-block:: python

    from evaluation_metrics.diversity_metrics import Topic_diversity
    topic_diversity = Topic_diversity(td_parameters) # Initialize metric
    topic_diversity_score = topic_diversity.score(model_output) # Compute score of the metric
    
Optimize a model
----------------

To optimize a model you need to select a model, a metric and the hyperparameters to optimize.

First choose the model.

.. code-block:: python

    from models.LDA import LDA_Model
    model = LDA_Model(dataset)

Choose the metric.

.. code-block:: python
    
    from evaluation_metrics.diversity_metrics import Topic_diversity
    topic_diversity = Topic_diversity(td_parameters) # Initialize metric

Define the search space of the hyperparameters to optimize.

.. code-block:: python

    search_space = {
    "alpha": Real(low=0.001, high=5.0),
    "eta": Real(low=0.001, high=5.0)
    }
    
Initialize an optimizer object and start the optimization.

.. code-block:: python

    from optimization.optimizer import Optimizer
    optimizer = Optimizer(
    model,
    topic_diversity,
    search_space)
    result = optimizer.optimize()
    
The result will be an object with optimized hyperparameters, best value of the metric and hyperparameters and metric value for each iteration of the optimization, to visualize the informations you can use the plot and plot_all methods of the result.

.. code-block:: python
    result.plot_all()
