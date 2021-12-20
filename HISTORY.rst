=======
History
=======

1.10.2 (2021-12-20)
--------------------
* Bug fix optimizer evaluation with additional metrics (#46)

1.10.1 (2021-12-08)
--------------------
* Bug fix Coherence with word embeddings (#43, #45)

1.10.0 (2021-11-21)
--------------------
* ETM now supports different formats of word embeddings (#36)
* Bug fix similarity measures (#41)
* Minor fixes

1.9.0 (2021-09-27)
------------------
* Bug fix preprocessing (#26)
* Bug fix ctm (#28)
* Bug fix weirbo_centroid (#31)
* Added new Italian datasets
* Minor fixes

1.8.3 (2021-07-26)
------------------
* Gensim migration from 3.8 to >=4.0.0

1.8.2 (2021-07-25)
------------------
* Fixed unwanted sorting of documents

1.8.1 (2021-07-08)
------------------
* Fixed gensim version (#22)

1.8.0 (2021-06-18)
------------------
* Added per-topic kl-uniform significance


1.7.1 (2021-06-09)
------------------
* Handling multilabel classification
* Fixed preprocessing when dataset is not split (#17)

1.6.0 (2021-05-20)
------------------
* Added regularization hyperparameter to NMF_scikit
* Added similarity metrics
* Fixed handling of stopwords in preprocessing
* Fixed coherence and diversity metrics
* Added new metrics tests

1.4.0 (2021-05-12)
------------------
* Fixed CTM training when only training dataset is used
* Dashboard bugs fixed
* Minor bug fixes
* Added new tests for TM training

1.3.0 (2021-04-25)
------------------
* Added parameter num_samples to CTM, NeuralLDA and ProdLDA
* Bug fix AVITM

1.2.1 (2021-04-21)
------------------
* Bug fix info dataset

1.2.0 (2021-04-20)
------------------
* Tomotopy LDA's implementation should work now

1.1.1 (2021-04-19)
------------------
* bug fix dataset download
* CTM is no longer verbose


1.1.0 (2021-04-18)
------------------
* New classification metrics
* Vocabulary downloader fix

1.0.2 (2021-04-16)
------------------
* Dataset downloader fix

1.0.0 (2021-04-16)
------------------
* New metrics initialization (do not support dictionaries as input anymore)
* Optimization, dataset and dashboard bug fixes
* Refactoring
* Updated README and documentation

0.4.0 (2021-04-15)
------------------
* Dataset preprocessing produces also an indexes.txt file containing the indexes of the documents
* Eval metrics bug fixes
* BBC news added in the correct format

0.3.0 (2021-04-10)
------------------
* Bug fixes

0.2.0 (2021-03-30)
------------------

* New dataset format


0.1.0 (2021-03-11)
------------------

* First release on PyPI.
