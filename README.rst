=======
OCTIS
=======

.. |colab1| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/MIND-Lab/OCTIS/blob/master/examples/models/LDA_training_only.ipynb
    :alt: Open In Colab

.. |colab2| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/MIND-Lab/OCTIS/blob/master/examples/optimization/optimizing_ETM.ipynb
    :alt: Open In Colab

.. |colab3| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/MIND-Lab/OCTIS/blob/master/examples/optimization/optimizing_LDA.ipynb
    :alt: Open In Colab

.. image:: https://img.shields.io/pypi/v/octis.svg
        :target: https://pypi.python.org/pypi/octis

.. image:: https://github.com/MIND-Lab/OCTIS/workflows/Python%20package/badge.svg
        :target: https://github.com/MIND-Lab/OCTIS/actions

.. image:: https://readthedocs.org/projects/octis/badge/?version=latest
        :target: https://octis.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/MIND-Lab/OCTIS/blob/master/examples/optimization/optimizing_ETM.ipynb
        :alt: Open In Colab

.. image:: https://github.com/MIND-Lab/OCTIS/blob/master/logo.png?raw=true
  :width: 100
  :alt: Logo

OCTIS (Optimizing and Comparing Topic models Is Simple) aims at training, analyzing and comparing
Topic Models, whose optimal hyper-parameters are estimated by means of a Bayesian Optimization approach.

Install
--------

You can install OCTIS with the following command:
::

    pip install octis

You can find the requirements in the `requirements.txt` file.


Features
--------

* We provide a set of state-of-the-art preprocessed text datasets (or you can preprocess your own dataset)
* We provide a set of well-known topic models (both classical and neurals), or you can integrate your own model
* You can evaluate your model using several state-of-the-art evaluation metrics
* You can optimize the hyperparameters of the models with respect to a given metric using Bayesian Optimization
* We provide a simple web dashboard for starting and controlling the optimization experiments


Get a preprocessed dataset
--------------------------

To acquire a dataset you can use one of the built-in sources.

.. code-block:: python

   from octis.dataset.dataset import Dataset
    dataset = Dataset()
    dataset.load("octis/preprocessed_datasets/m10")

Or use your own.

.. code-block:: python

    import octis.preprocessing.sources.custom_dataset as source
    dataset = source.retrieve("path\to\dataset")


A custom dataset must have a document for each line of the file.
Datasets can be partitioned in train and test sets.

Preprocess
----------

To preprocess a dataset Initialize a Pipeline_handler and use the preprocess method.

.. code-block:: python

    from octis.preprocessing.pipeline_handler import Pipeline_handler

    pipeline_handler = Pipeline_handler(dataset) # Initialize pipeline handler
    preprocessed = pipeline_handler.preprocess() # preprocess

    preprocessed.save("dataset_folder") # Save the preprocessed dataset


For the customization of the preprocess pipeline see the optimization demo example in the examples folder.

Train a model
-------------

To build a model, load a preprocessed dataset, customize the model hyperparameters and use the train_model() method of the model class.

.. code-block:: python

    from octis.dataset.dataset import Dataset
    from octis.models.LDA import LDA

    # Load a dataset
    dataset = Dataset()
    dataset.load("dataset_folder")

    model = LDA(num_topics=25)  # Create model
    model_output = model.train_model(dataset) # Train the model


If the dataset is partitioned, you can choose to:

* Train the model on the training set and test it on the test documents
* Train the model on the training set and update it with the test set
* Train the model with the whole dataset, regardless of any partition.

Evaluate a model
----------------

To evaluate a model, choose a metric and use the score() method of the metric class.

.. code-block:: python

    from octis.evaluation_metrics.diversity_metrics import TopicDiversity

    # Set metric parameters
    td_parameters ={'topk':10}

    metric = TopicDiversity(td_parameters) # Initialize metric
    topic_diversity_score = metric.score(model_output) # Compute score of the metric


Optimize a model
----------------

To optimize a model you need to select a dataset, a metric and the search space of the hyperparameters to optimize.

.. code-block:: python

    from octis.optimization.optimizer import Optimizer

    search_space = {
    "alpha": Real(low=0.001, high=5.0),
    "eta": Real(low=0.001, high=5.0)
    }

    number_of_call=5
    model_runs=3
    save_path="results"
    # Initialize an optimizer object and start the optimization.
    optimizer=Optimizer()
    OptObject=optimizer.optimize(model,dataset, npmi,search_space,
                                    number_of_call=number_of_call,
                                    model_runs=model_runs,
                                    save_path=save_path)
    #save the results of th optimization in a csv file
    OptObject.save_to_csv("results.csv")

The result will provide best-seen value of the metric with the corresponding hyperparameter configuration, and the hyperparameters and metric value for each iteration of the optimization. To visualize this information, you have to set 'plot' attribute of Bayesian_optimization to True.

You can find more here: `optimizer README`_

Examples and Tutorials
-----------------------

Our Colab Tutorials:

+--------------------------------------------------------------------------------+------------------+
| Name                                                                           | Link             |
+================================================================================+==================+
| How to build a topic model and evaluate the results.                           | |colab1|         |
+--------------------------------------------------------------------------------+------------------+
| Optimizing a topic model (Example with ETM and 20Newsgroup)                    | |colab2|         |
+--------------------------------------------------------------------------------+------------------+
| Optimizing a topic model (Example with LDA and M10)                            | |colab3|         |
+--------------------------------------------------------------------------------+------------------+

Available Models
----------------

* AVITM
* CTM
* ETM
* HDP
* LDA
* LSI
* NMF
* NeuralLDA
* ProdLDA

Available Datasets
-------------------

* 20Newsgroup
* BBC News
* DBLP
* M10

Disclaimer
~~~~~~~~~~~~~

Similarly to `TensorFlow Datasets`_ and HuggingFace's `nlp`_ library, we just downloaded and prepared public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license and to cite the right owner of the dataset.

If you're a dataset owner and wish to update any part of it, or do not want your dataset to be included in this library, please get in touch through a GitHub issue.

If you're a dataset owner and wish to include your dataset in this library, please get in touch through a GitHub issue.

Implement your own Model
------------------------

Models inherit from the class `Abstract_Model` defined in `models/model.py` .
To build your own model your class must override the `train_model(self, dataset, hyperparameters)` method which always require at least a `Dataset` object and a `Dictionary` of hyperparameters as input and should return a dictionary with the output of the model as output.

To better understand how a model work, let's have a look at the LDA implementation.
The first step in developing a custom model is to define the dictionary of default hyperparameters values:

.. code-block:: python

    hyperparameters = {'corpus': None, 'num_topics': 100,
        'id2word': None, 'alpha': 'symmetric',
        'eta': None, # ...
        'callbacks': None}

Defining the default hyperparameters values allows users to work on a subset of them without having to assign a value to each parameter.

The following step is the `train_model()` override:

.. code-block:: python

    def train_model(self, dataset, hyperparameters={}, top_words=10):

The LDA method requires a dataset, the hyperparameters dictionary and an extra (optional) argument used to select how many of the most significative words track for each topic.

With the hyperparameters defaults, the ones in input and the dataset you should be able to write your own code and return as output a dictionary with at least 3 entries:

* `topics`: the list of the most significative words foreach topic (list of lists of strings).
* `topic-word-matrix`: an NxV matrix of weights where N is the number of topics and V is the vocabulary length.
* `topic-document-matrix`: an NxD matrix of weights where N is the number of topics and D is the number of documents in the corpus.

if your model support the training/test partitioning it should also return:

* `test-topic-document-matrix`: the document topic matrix of the test set.

In case the model isn't updated with the test set.
Or:

* `test-topics`: the list of the most significative words foreach topic (list of lists of strings) of the model updated with the test set.
* `test-topic-word-matrix`: an NxV matrix of weights where N is the number of topics and V is the vocabulary length of the model updated with the test set.
* `test-topic-document-matrix`: an NxD matrix of weights where N is the number of topics and D is the number of documents in the corpus of the model updated with the test set.

If the model is updated with the test set.

Dashboard
---------

OCTIS includes a user friendly graphical interface for creating, monitoring and viewing experiments.
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

Team
------

Project and Development Lead
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `Silvia Terragni`_ <s.terragni4@campus.unimib.it>
* Elisabetta Fersini <elisabetta.fersini@unimib.it>
* Antonio Candelieri <antonio.candelieri@unimib.it>

Current Contributors
~~~~~~~~~~~~~~~~~~~~~~

* Pietro Tropeano <p.tropeano1@campus.unimib.it> Framework architecture, Preprocessing, Topic Models, Evaluation metrics and Web Dashboard
* Bruno Galuzzi <bruno.galuzzi@unimib.it> Bayesian Optimization
* Silvia Terragni <s.terragni4@campus.unimib.it> Overall project

Past Contributors
~~~~~~~~~~~~~~~~~~~~
* Lorenzo Famiglini <l.famiglini@campus.unimib.it> Neural models integration
* Davide Pietrasanta <d.pietrasanta@campus.unimib.it> Bayesian Optimization

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _Silvia Terragni: https://silviatti.github.io/
.. _Optimizer README: https://github.com/MIND-Lab/topic-modeling-evaluation-framework/blob/develop-package/octis/optimization/README.md
.. _TensorFlow Datasets: https://github.com/tensorflow/datasets
.. _nlp: https://github.com/huggingface/nlp
