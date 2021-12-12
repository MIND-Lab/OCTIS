=========================================================
OCTIS : Optimizing and Comparing Topic Models is Simple!
=========================================================


.. |colab1| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/MIND-Lab/OCTIS/blob/master/examples/OCTIS_LDA_training_only.ipynb
    :alt: Open In Colab

.. |colab2| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/MIND-Lab/OCTIS/blob/master/examples/OCTIS_Optimizing_CTM.ipynb
    :alt: Open In Colab
.. |twitter_silvia| image:: https://img.shields.io/twitter/follow/TerragniSilvia?style=social
    :target: https://twitter.com/intent/follow?screen_name=TerragniSilvia
    :alt: Follow TerragniSilvia on Twitter
.. |twitter_betta| image:: https://img.shields.io/twitter/follow/FersiniE?style=social
    :target: https://twitter.com/intent/follow?screen_name=FersiniE
    :alt: Follow FersiniE on Twitter

.. image:: https://img.shields.io/pypi/v/octis.svg
        :target: https://pypi.python.org/pypi/octis

.. image:: https://github.com/MIND-Lab/OCTIS/workflows/Python%20package/badge.svg
        :target: https://github.com/MIND-Lab/OCTIS/actions

.. image:: https://readthedocs.org/projects/octis/badge/?version=latest
        :target: https://octis.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/github/contributors/MIND-Lab/OCTIS
        :target: https://github.com/MIND-Lab/OCTIS/graphs/contributors/
        :alt: Contributors

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
        :target: https://lbesson.mit-license.org/
        :alt: License

.. image:: https://img.shields.io/github/stars/mind-lab/OCTIS?logo=github
        :target: https://github.com/mind-lab/OCTIS/stargazers
        :alt: Github Stars
       
.. image:: https://pepy.tech/badge/octis/month
        :target: https://pepy.tech/project/octis
        :alt: Monthly Downloads
        
.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/MIND-Lab/OCTIS/blob/master/examples/OCTIS_Optimizing_CTM.ipynb
        :alt: Open In Colab
      

.. image:: https://github.com/MIND-Lab/OCTIS/blob/master/logo.png?raw=true
  :width: 100
  :alt: Logo
  
  



**OCTIS (Optimizing and Comparing Topic models Is Simple)** aims at training, analyzing and comparing
Topic Models, whose optimal hyperparameters are estimated by means of a Bayesian Optimization approach. This work has been accepted to the demo track of EACL2021. `Click to read the paper`_!

.. contents:: Table of Contents 
   :depth: 2

***************
Install
***************


You can install OCTIS with the following command:
::

    pip install octis

You can find the requirements in the `requirements.txt` file.

***************
Main Features
***************


* Preprocess your own dataset or use one of the already-preprocessed benchmark datasets
* Well-known topic models (both classical and neurals)
* Evaluate your model using different state-of-the-art evaluation metrics
* Optimize the models' hyperparameters for a given metric using Bayesian Optimization
* Python library for advanced usage or simple web dashboard for starting and controlling the optimization experiments


***********************
Examples and Tutorials
***********************

To easily understand how to use OCTIS, we invite you to try our tutorials out :)

+--------------------------------------------------------------------------------+------------------+
| Name                                                                           | Link             |
+================================================================================+==================+
| How to build a topic model and evaluate the results (LDA on 20Newsgroups)      | |colab1|         |
+--------------------------------------------------------------------------------+------------------+
| How to optimize the hyperparameters of a neural topic model (CTM on M10)       | |colab2|         |
+--------------------------------------------------------------------------------+------------------+


**************************
Datasets and Preprocessing
**************************

Load a preprocessed dataset
============================

To load one of the already preprocessed datasets as follows:

.. code-block:: python

   from octis.dataset.dataset import Dataset
   dataset = Dataset()
   dataset.fetch_dataset("20NewsGroup")

Just use one of the dataset names listed below. Note: it is case-sensitive!

Available Datasets
============================

+--------------+--------------+--------+---------+----------+----------+
|Name in OCTIS | Source       | # Docs | # Words | # Labels | Language |
+==============+==============+========+=========+==========+==========+
| 20NewsGroup  | 20Newsgroup_ |  16309 |    1612 |       20 | English  |
+--------------+--------------+--------+---------+----------+----------+
| BBC_News     | BBC-News_    |   2225 |    2949 |        5 | English  |
+--------------+--------------+--------+---------+----------+----------+
| DBLP         | DBLP_        |  54595 |    1513 |        4 | English  |
+--------------+--------------+--------+---------+----------+----------+
| M10          | M10_         |   8355 |    1696 |       10 | English  |
+--------------+--------------+--------+---------+----------+----------+
| DBPedia_IT   | DBPedia_IT_  |   4251 |    2047 |        5 | Italian  |
+--------------+--------------+--------+---------+----------+----------+
| Europarl_IT  | Europarl_IT_ |   3613 |    2000 |       NA | Italian  |
+--------------+--------------+--------+---------+----------+----------+

.. _20Newsgroup: https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
.. _BBC-News: https://github.com/MIND-Lab/OCTIS
.. _DBLP: https://dblp.org/rec/conf/ijcai/PanWZZW16.html?view=bibtex
.. _M10: https://dblp.org/rec/conf/ijcai/PanWZZW16.html?view=bibtex
.. _DBPedia_IT: https://www.dbpedia.org/resources/ontology/
.. _Europarl_IT: https://www.statmt.org/europarl/

Load a Custom Dataset
============================
Otherwise, you can load a custom preprocessed dataset in the following way:

.. code-block:: python

   from octis.dataset.dataset import Dataset
   dataset = Dataset()
   dataset.load_custom_dataset_from_folder("../path/to/the/dataset/folder")

Make sure that the dataset is in the following format:
    * corpus file: a .tsv file (tab-separated) that contains up to three columns, i.e. the document, the partitition, and the label associated to the document (optional).
    * vocabulary: a .txt file where each line represents a word of the vocabulary

The partition can be "train" for the training partition, "test" for testing partition, or "val" for the validation partition. An example of dataset can be found here: `sample_dataset`_.

Disclaimer
~~~~~~~~~~~~~

Similarly to `TensorFlow Datasets`_ and HuggingFace's `nlp`_ library, we just downloaded and prepared public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license and to cite the right owner of the dataset.

If you're a dataset owner and wish to update any part of it, or do not want your dataset to be included in this library, please get in touch through a GitHub issue.

If you're a dataset owner and wish to include your dataset in this library, please get in touch through a GitHub issue.

Preprocess a Dataset
============================

To preprocess a dataset, import the preprocessing class and use the preprocess_dataset method.

.. code-block:: python


    import os
    import string
    from octis.preprocessing.preprocessing import Preprocessing
    os.chdir(os.path.pardir)

    # Initialize preprocessing
    preprocessor = Preprocessing(vocabulary=None, max_features=None, 
                                 remove_punctuation=True, punctuation=string.punctuation,
                                 lemmatize=True, stopword_list='english',
                                 min_chars=1, min_words_docs=0)
    # preprocess
    dataset = preprocessor.preprocess_dataset(documents_path=r'..\corpus.txt', labels_path=r'..\labels.txt')

    # save the preprocessed dataset
    dataset.save('hello_dataset')


For more details on the preprocessing see the preprocessing demo example in the examples folder.


*****************************
Topic Models and Evaluation
*****************************

Train a model
==============

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

Available Models
=================

+-------------------------------------------+-----------------------------------------------------------+
| Name                                      | Implementation                                            |
+===========================================+===========================================================+
| CTM `(Bianchi et al. 2021)`_              | https://github.com/MilaNLProc/contextualized-topic-models |
+-------------------------------------------+-----------------------------------------------------------+
| ETM `(Dieng et al. 2020)`_                | https://github.com/adjidieng/ETM                          |
+-------------------------------------------+-----------------------------------------------------------+
| HDP `(Blei et al. 2004)`_                 | https://radimrehurek.com/gensim/                          |
+-------------------------------------------+-----------------------------------------------------------+
| LDA `(Blei et al. 2003)`_                 | https://radimrehurek.com/gensim/                          |
+-------------------------------------------+-----------------------------------------------------------+
| LSI `(Landauer et al. 1998)`_             | https://radimrehurek.com/gensim/                          |
+-------------------------------------------+-----------------------------------------------------------+
| NMF `(Lee and Seung 2000)`_               | https://radimrehurek.com/gensim/                          |
+-------------------------------------------+-----------------------------------------------------------+
| NeuralLDA `(Srivastava and Sutton 2017)`_ | https://github.com/estebandito22/PyTorchAVITM             |
+-------------------------------------------+-----------------------------------------------------------+
| ProdLda `(Srivastava and Sutton 2017)`_   | https://github.com/estebandito22/PyTorchAVITM             |
+-------------------------------------------+-----------------------------------------------------------+


.. _(Bianchi et al. 2021): https://www.aclweb.org/anthology/2021.eacl-main.143/
.. _(Dieng et al. 2020): https://www.aclweb.org/anthology/2020.tacl-1.29 
.. _(Blei et al. 2004): https://people.eecs.berkeley.edu/~jordan/papers/hdp.pdf
.. _(Blei et al. 2003): https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
.. _(Landauer et al. 1998): http://lsa.colorado.edu/papers/dp1.LSAintro.pdf
.. _(Lee and Seung 2000): https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization
.. _(Srivastava and Sutton 2017): https://arxiv.org/abs/1703.01488 

If you use one of these implementations, make sure to cite the right paper.

If you implemented a model and wish to update any part of it, or do not want your model to be included in this library, please get in touch through a GitHub issue.

If you implemented a model and wish to include your model in this library, please get in touch through a GitHub issue. Otherwise, if you want to include the model by yourself, see the following section.

Evaluate a model
==================

To evaluate a model, choose a metric and use the :code:`score()` method of the metric class.

.. code-block:: python

    from octis.evaluation_metrics.diversity_metrics import TopicDiversity

    metric = TopicDiversity(topk=10) # Initialize metric
    topic_diversity_score = metric.score(model_output) # Compute score of the metric

Available metrics
==================

* **Classification Metrics**:

    * F1-score_ : :code:`F1Score(dataset)`
    * Precision_ : :code:`PrecisionScore(dataset)`
    * Recall_ : :code:`RecallScore(dataset)`
    * Accuracy_ : :code:`AccuracyScore(dataset)`

.. _F1-score: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/classification_metrics.py#L117
.. _Precision: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/classification_metrics.py#L145
.. _Recall: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/classification_metrics.py#L171
.. _Accuracy: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/classification_metrics.py#L196

* **Coherence Metrics**:

    * `UMass Coherence`_ : :code:`Coherence({'measure':'c_umass'}`
    * `C_V Coherence`_ : :code:`Coherence({'measure':'c_v'}`
    * `UCI Coherence`_ : :code:`Coherence({'measure':'c_uci'}`
    * `NPMI Coherence`_ : :code:`Coherence({'measure':'c_npmi'}`
    * `Word Embedding-based Coherence Pairwise`_ : :code:`WECoherencePairwise()`
    * `Word Embedding-based Coherence Centroid`_ : :code:`WECoherenceCentroid()`

.. _`UMass Coherence`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/coherence_metrics.py#L15
.. _`C_V Coherence`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/coherence_metrics.py#L15
.. _`UCI Coherence`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/coherence_metrics.py#L15
.. _`NPMI Coherence`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/coherence_metrics.py#L15
.. _`Word Embedding-based Coherence Pairwise`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/coherence_metrics.py#L67
.. _`Word Embedding-based Coherence Centroid`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/coherence_metrics.py#L126

* **Diversity Metrics**:

    * `Topic Diversity`_ : :code:`TopicDiversity()`
    * `InvertedRBO`_ : :code:`InvertedRBO()`
    * `Word Embedding-based InvertedRBO Matches`_ : :code:`WordEmbeddingsInvertedRBO()`
    * `Word Embedding-based InvertedRBO Centroid`_ : :code:`WordEmbeddingsInvertedRBOCentroid()`
    * `Log odds ratio`_ : :code:`LogOddsRatio()`
    * `Kullback-Liebler Divergence`_ : :code:`KLDivergence()`

.. _`Topic Diversity`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/diversity_metrics.py#L12
.. _`InvertedRBO`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/diversity_metrics.py#L56
.. _`Word Embedding-based InvertedRBO Matches`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/diversity_metrics.py#L92
.. _`Word Embedding-based InvertedRBO Centroid`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/diversity_metrics.py#L147
.. _`Log odds ratio`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/diversity_metrics.py#L184
.. _`Kullback-Liebler Divergence`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/diversity_metrics.py#L209

* **Similarity Metrics**:

    * `Ranked-Biased Overlap`_ : :code:`RBO()`
    * `Word Embedding-based RBO Matches`_ : :code:`WordEmbeddingsRBOMatch()`
    * `Word Embedding-based RBO Centroid`_ : :code:`WordEmbeddingsRBOCentroid()`
    * `Word Embeddings-based Pairwise Similarity`_ : :code:`WordEmbeddingsPairwiseSimilarity()`
    * `Word Embeddings-based Centroid Similarity`_ : :code:`WordEmbeddingsCentroidSimilarity()`
    * `Word Embeddings-based Weighted Sum Similarity`_ : :code:`WordEmbeddingsWeightedSumSimilarity()`
    * `Pairwise Jaccard Similarity`_ : :code:`PairwiseJaccardSimilarity()`


.. _`Word Embedding-based RBO Matches`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/similarity_metrics.py#L11
.. _`Word Embedding-based RBO Centroid`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/similarity_metrics.py#L35
.. _`Word Embeddings-based Pairwise Similarity`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/similarity_metrics.py#L59
.. _`Word Embeddings-based Centroid Similarity`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/similarity_metrics.py#L103
.. _`Ranked-Biased Overlap`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/similarity_metrics.py#L201
.. _`Word Embeddings-based Weighted Sum Similarity`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/similarity_metrics.py#L158
.. _`Pairwise Jaccard Similarity`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/similarity_metrics.py#L223
 

* **Topic significance Metrics**:

    * `KL Uniform`_ : :code:`KL_uniform()`
    * `KL Vacuous`_ : :code:`KL_vacuous()`
    * `KL Background`_ : :code:`KL_background()`
    
.. _`KL Uniform`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/topic_significance_metrics.py#L37
.. _`KL Vacuous`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/topic_significance_metrics.py#L84
.. _`KL Background`: https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/topic_significance_metrics.py#L138
 

Implement your own Model
=========================

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



*****************************
Hyperparameter Optimization
*****************************

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


*****************************
Dashboard
*****************************


OCTIS includes a user friendly graphical interface for creating, monitoring and viewing experiments.
Following the implementation standards of datasets, models and metrics the dashboard will automatically update and allow you to use your own custom implementations.

To run rhe dashboard you need to clone the repo.
While in the project directory run the following command:

.. code-block:: bash

    python OCTIS/dashboard/server.py


The browser will open and you will be redirected to the dashboard.
In the dashboard you can:

* Create new experiments organized in batch
* Visualize and compare all the experiments
* Visualize a custom experiment
* Manage the experiment queue


*****************************
How to cite our work
*****************************
This work has been accepted at the demo track of EACL 2021! `Click to read the paper`_!
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

*****************************
Team
*****************************

Project and Development Lead
=============================

- `Silvia Terragni`_ <s.terragni4@campus.unimib.it> |twitter_silvia|
- Elisabetta Fersini <elisabetta.fersini@unimib.it> |twitter_betta|
- Antonio Candelieri <antonio.candelieri@unimib.it>



Current Contributors
=============================

- Pietro Tropeano <p.tropeano1@campus.unimib.it> Framework architecture, Preprocessing, Topic Models, Evaluation metrics and Web Dashboard
- Bruno Galuzzi <bruno.galuzzi@unimib.it> Bayesian Optimization
- Silvia Terragni <s.terragni4@campus.unimib.it> Overall project

Past Contributors
=============================

* Lorenzo Famiglini <l.famiglini@campus.unimib.it> Neural models integration
* Davide Pietrasanta <d.pietrasanta@campus.unimib.it> Bayesian Optimization



*****************************
Credits
*****************************

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template. Thanks to all the developers that released their topic models' implementations. A special thanks goes to tenggaard_ who helped us find many bugs in early octis releases :) 

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`Click to read the paper`: https://www.aclweb.org/anthology/2021.eacl-demos.31/
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _Silvia Terragni: https://silviatti.github.io/
.. _sample_dataset: https://github.com/MIND-Lab/OCTIS/tree/master/preprocessed_datasets/sample_dataset
.. _Optimizer README: https://github.com/MIND-Lab/topic-modeling-evaluation-framework/blob/develop-package/octis/optimization/README.md
.. _TensorFlow Datasets: https://github.com/tensorflow/datasets
.. _nlp: https://github.com/huggingface/nlp
.. _tenggaard: https://github.com/tenggaard
