=========================================================
OCTIS : Optimizing and Comparing Topic Models is Simple!
=========================================================

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


.. image:: https://img.shields.io/github/contributors/MIND-Lab/OCTIS
        :target: https://github.com/MIND-Lab/OCTIS/graphs/contributors/
        :alt: Contributors

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
        :target: https://lbesson.mit-license.org/
        :alt: License

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

* Preprocess your own dataset or use one of the already-preprocessed benchmark datasets
* Well-known topic models (both classical and neurals)
* Evaluate your model using different state-of-the-art evaluation metrics
* Optimize the models' hyperparameters for a given metric using Bayesian Optimization
* Python library for advanced usage or simple web dashboard for starting and controlling the optimization experiments


Examples and Tutorials
-----------------------

To easily understand how to use OCTIS, we invite you to try our tutorials out :)

+---------------------------------------------------------------------------------+------------------+
| Name                                                                            | Link             |
+=================================================================================+==================+
| How to train a topic model and evaluate their results                           | |colab1|         |
+---------------------------------------------------------------------------------+------------------+
| Optimizing a neural topic model (Example with ETM on 20Newsgroup)               | |colab2|         |
+---------------------------------------------------------------------------------+------------------+
| Optimizing a classical topic model (Example with LDA on M10)                    | |colab3|         |
+---------------------------------------------------------------------------------+------------------+



Load a preprocessed dataset
----------------------------

To load one of the already preprocessed datasets as follows:

.. code-block:: python

   from octis.dataset.dataset import Dataset
   dataset = Dataset()
   dataset.fetch_dataset("20NewsGroup")

Just use one of the dataset names listed below. Note: it is case-sensitive! 

Available Datasets
-------------------

+--------------+--------------+--------+---------+----------+
| Name         | Source       | # Docs | # Words | # Labels |
+==============+==============+========+=========+==========+
| 20NewsGroup  | 20Newsgroup_ |  16309 |    1612 |       20 |
+--------------+--------------+--------+---------+----------+
| BBC_News     | BBC-News_    |   2225 |    2949 |        5 |
+--------------+--------------+--------+---------+----------+
| DBLP         | DBLP_        |  54595 |    1513 |        4 |
+--------------+--------------+--------+---------+----------+
| M10          | M10_         |   8355 |    1696 |       10 |
+--------------+--------------+--------+---------+----------+

.. _20Newsgroup: https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
.. _BBC-News: https://github.com/MIND-Lab/OCTIS
.. _DBLP: https://dblp.org/rec/conf/ijcai/PanWZZW16.html?view=bibtex
.. _M10: https://dblp.org/rec/conf/ijcai/PanWZZW16.html?view=bibtex

Otherwise, you can load a custom preprocessed dataset in the following way:

.. code-block:: python

   from octis.dataset.dataset import Dataset
   dataset = Dataset()
   dataset.load_custom_dataset_from_folder("../path/to/the/dataset/folder")

Make sure that the dataset is in the following format:
    * corpus file: a .tsv file (tab-separated) that contains up to three columns, i.e. the document, the partitition, and the label associated to the document (optional).
    * vocabulary: a .txt file where each line represents a word of the vocabulary

The partition can be "training", "test" or "validation". An example of dataset can be found here: `sample_dataset_`.

Disclaimer
~~~~~~~~~~~~~

Similarly to `TensorFlow Datasets`_ and HuggingFace's `nlp`_ library, we just downloaded and prepared public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license and to cite the right owner of the dataset.

If you're a dataset owner and wish to update any part of it, or do not want your dataset to be included in this library, please get in touch through a GitHub issue.

If you're a dataset owner and wish to include your dataset in this library, please get in touch through a GitHub issue.

Preprocess
-----------

To preprocess a dataset, import the preprocessing class and use the preprocess_dataset method.

.. code-block:: python


    import os
    import string
    from octis.preprocessing.preprocessing import Preprocessing
    os.chdir(os.path.pardir)

    # Initialize preprocessing
    p = Preprocessing(vocabulary=None, max_features=None, remove_punctuation=True, punctuation=string.punctuation,
                      lemmatize=True, remove_stopwords=True, stopword_list=['am', 'are', 'this', 'that'],
                      min_chars=1, min_words_docs=0)
    # preprocess
    dataset = p.preprocess_dataset(documents_path=r'..\corpus.txt', labels_path=r'..\labels.txt')

    # save the preprocessed dataset
    dataset.save('hello_dataset')


For more details on the preprocessing see the preprocessing demo example in the examples folder.

Train a model
--------------

To build a model, load a preprocessed dataset, set the model hyperparameters and use :code:`train_model()` to train the model.

.. code-block:: python

    from octis.dataset.dataset import Dataset
    from octis.models.LDA import LDA

    # Load a dataset
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder("dataset_folder")

    model = LDA(num_topics=25)  # Create model
    model_output = model.train_model(dataset) # Train the model


If the dataset is partitioned, you can:

* Train the model on the training set and test it on the test documents
* Train the model with the whole dataset, regardless of any partition.

Evaluate a model
----------------

To evaluate a model, choose a metric and use the :code:`score()` method of the metric class.

.. code-block:: python

    from octis.evaluation_metrics.diversity_metrics import TopicDiversity
    
    metric = TopicDiversity(topk=10) # Initialize metric
    topic_diversity_score = metric.score(model_output) # Compute score of the metric

Available metrics
-----------------

Classification Metrics:

* F1 measure (:code:`F1Score()`)

Coherence Metrics:

* UMass Coherence (:code:`Coherence({'measure':'c_umass'}`)
* C_V Coherence (:code:`Coherence({'measure':'c_v'}`)
* UCI Coherence (:code:`Coherence({'measure':'c_uci'}`)
* NPMI Coherence (:code:`Coherence({'measure':'c_npmi'}`)
* Word Embedding-based Coherence Pairwise (:code:`WECoherencePairwise()`)
* Word Embedding-based Coherence Centroid (:code:`WECoherenceCentroid()`)

Diversity Metrics:

* Topic Diversity (:code:`TopicDiversity()`)
* InvertedRBO (:code:`InvertedRBO()`)
* Word Embedding-based InvertedRBO (:code:`WordEmbeddingsInvertedRBO()`)
* Word Embedding-based InvertedRBO centroid (:code:`WordEmbeddingsInvertedRBOCentroid()`)

Topic significance Metrics:

* KL Uniform (:code:`KL_uniform()`)
* KL Vacuous (:code:`KL_vacuous()`)
* KL Background (:code:`KL_background()`)


Optimize a model
----------------

To optimize a model you need to select a dataset, a metric and the search space of the hyperparameters to optimize. 
For the types of the hyperparameters, we use :code:`scikit-optimize` types (https://scikit-optimize.github.io/stable/modules/space.html)

.. code-block:: python

    from octis.optimization.optimizer import Optimizer
    from skopt.space.space import Real

    # Define the search space. To see which hyperparameters to optimize, see the topic model's initialization signature
    search_space = {"alpha": Real(low=0.001, high=5.0), "eta": Real(low=0.001, high=5.0)}

    # Initialize an optimizer object and start the optimization.
    optimizer=Optimizer()
    optResult=optimizer.optimize(model, dataset, eval_metric, search_space, save_path="../results" # path to store the results
                                 number_of_call=30, # number of optimization iterations
                                 model_runs=5) # number of runs of the topic model 
    #save the results of th optimization in a csv file
    optResult.save_to_csv("results.csv")

The result will provide best-seen value of the metric with the corresponding hyperparameter configuration, and the hyperparameters and metric value for each iteration of the optimization. To visualize this information, you have to set 'plot' attribute of Bayesian_optimization to True.

You can find more here: `optimizer README`_


Available Models
----------------

+--------------------------------+-----------------------------------------------------------+
| Name                           | Implementation                                            |
+================================+===========================================================+
| CTM (Bianchi et al. 2020)      | https://github.com/MilaNLProc/contextualized-topic-models |
+--------------------------------+-----------------------------------------------------------+
| ETM (Dieng et al. 2019)        | https://github.com/adjidieng/ETM                          |
+--------------------------------+-----------------------------------------------------------+
| HDP (Blei et al. 2004)         | https://radimrehurek.com/gensim/                          |
+--------------------------------+-----------------------------------------------------------+
| LDA (Blei et al. 2001)         | https://radimrehurek.com/gensim/                          |
+--------------------------------+-----------------------------------------------------------+
| LSI (Deerwester et al. 2009)   | https://radimrehurek.com/gensim/                          |
+--------------------------------+-----------------------------------------------------------+
| NMF (Lee and Seung 2000)       | https://radimrehurek.com/gensim/                          |
+--------------------------------+-----------------------------------------------------------+
| NeuralLDA (Carrow et al. 2018) | https://github.com/estebandito22/PyTorchAVITM             |
+--------------------------------+-----------------------------------------------------------+
| ProdLda (Carrow et al. 2018)   | https://github.com/estebandito22/PyTorchAVITM             |
+--------------------------------+-----------------------------------------------------------+

If you use one of these implementations, make sure to cite the right paper.

If you implemented a model and wish to update any part of it, or do not want your model to be included in this library, please get in touch through a GitHub issue.

If you implemented a model and wish to include your model in this library, please get in touch through a GitHub issue. Otherwise, if you want to include the model by yourself, see the following section.

Implement your own Model
------------------------

Models inherit from the class `AbstractModel` defined in `octis/models/model.py` .
To build your own model your class must override the `train_model(self, dataset, hyperparameters)` method which always requires at least a `Dataset` object and a `Dictionary` of hyperparameters as input and should return a dictionary with the output of the model as output.

To better understand how a model work, let's have a look at the LDA implementation.
The first step in developing a custom model is to define the dictionary of default hyperparameters values:

.. code-block:: python

    hyperparameters = {'corpus': None, 'num_topics': 100, 'id2word': None, 'alpha': 'symmetric',
        'eta': None, # ...
        'callbacks': None}

Defining the default hyperparameters values allows users to work on a subset of them without having to assign a value to each parameter.

The following step is the `train_model()` override:

.. code-block:: python

    def train_model(self, dataset, hyperparameters={}, top_words=10):

The LDA method requires a dataset, the hyperparameters dictionary and an extra (optional) argument used to select how many of the most significative words track for each topic.

With the hyperparameters defaults, the ones in input and the dataset you should be able to write your own code and return as output a dictionary with at least 3 entries:

* *topics*: the list of the most significative words foreach topic (list of lists of strings).
* *topic-word-matrix*: an NxV matrix of weights where N is the number of topics and V is the vocabulary length.
* *topic-document-matrix*: an NxD matrix of weights where N is the number of topics and D is the number of documents in the corpus.

if your model supports the training/test partitioning it should also return:

* *test-topic-document-matrix*: the document topic matrix of the test set.

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


How to cite our work
---------------------
This work has been accepted at the demo track of EACL 2021! You can find it here: https://www.aclweb.org/anthology/2020.insights-1.5/ 
If you decide to use this resource, please cite:

::

    @inproceedings{terragni2020octis,
        title={{OCTIS}: Comparing and Optimizing Topic Models is Simple!},
        author={Terragni, Silvia and Fersini, Elisabetta and Galuzzi, Bruno Giovanni and Tropeano, Pietro and Candelieri, Antonio},
        year={2021},
        booktitle={Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations},
        month = apr,
        year = "2021",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/2021.eacl-demos.31",
        pages = "263--270",
    }



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

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template. Thanks to all the developers that released their topic models' implementations.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _Silvia Terragni: https://silviatti.github.io/
.. _sample_dataset: https://github.com/MIND-Lab/OCTIS/tree/master/preprocessed_datasets/sample_dataset
.. _Optimizer README: https://github.com/MIND-Lab/topic-modeling-evaluation-framework/blob/develop-package/octis/optimization/README.md
.. _TensorFlow Datasets: https://github.com/tensorflow/datasets
.. _nlp: https://github.com/huggingface/nlp
