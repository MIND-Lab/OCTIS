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

Load a preprocessed dataset
----------------------------

To load one of the already preprocessed datasets as follows:

.. code-block:: python

   from octis.dataset.dataset import Dataset
   dataset = Dataset()
   dataset.fetch_dataset("20NewsGroup")

Just use one of the dataset names listed above. Note: it is case-sensitive!


Load a custom preprocessed dataset
----------------------------

Otherwise, you can load a custom preprocessed dataset in the following way:

.. code-block:: python

   from octis.dataset.dataset import Dataset
   dataset = Dataset()
   dataset.load_custom_dataset_from_folder("../path/to/the/dataset/folder")

Make sure that the dataset is in the following format:
    * corpus file: a .tsv file (tab-separated) that contains up to three columns, i.e. the document, the partitition, and the label associated to the document (optional).
    * vocabulary: a .txt file where each line represents a word of the vocabulary

The partition can be "training", "test" or "validation". An example of dataset can be found here: `sample_dataset_`.
