# Topic Modeling Evaluation Framework

This framework aims to compare topic models' performance with respect to multiple different metrics. Topic models are optimized using Bayesian Optimization techniques.

Features
--------

* Provide a set of state-of-the-art preprocessed text datasets, or you can preprocess your own dataset
* Provide a set of known topic models, or you can integrate your own model
* Optimize models with respect to a given metric using Bayesian Optimization
* Evaluate your model using state-of-the-art evaluation metrics

Acquire dataset
---------------

To acquire a dataset you can use one of the built-in sources.

``` python
import preprocessing.sources.newsgroup as source
dataset = source.retrieve()
```

Or use your own.

``` python
import preprocessing.sources.custom_dataset as source
dataset = source.retrieve("path\to\dataset")
```

A custom dataset must have a document for each line of the file.
Datasets can be partitioned in train and test sets.

Preprocess
----------

To preprocess a dataset Initialize a Pipeline_handler and use the preprocess method.

``` python
from preprocessing.pipeline_handler import Pipeline_handler

pipeline_handler = Pipeline_handler(dataset) # Initialize pipeline handler
preprocessed = pipeline_handler.preprocess() # preprocess

preprocessed.save("dataset_folder") # Save the preprocessed dataset
```

For the customization of the preprocess pipeline see the optimization demo example in the examples folder.

Build a model
-------------

To build a model, load a preprocessed dataset, customize the model hyperparameters and use the train_model() method of the model class.

``` python
from dataset.dataset import Dataset
from models.LDA import LDA_Model

# Load a dataset
dataset = Dataset()
dataset.load("dataset_folder")

# Set hyperparameters
hyperparameters = {}
hyperparameters["num_topics"] = 20

model = LDA_Model()  # Create model
model_output = model.train_model(dataset, hyperparameters) # Train the model
```

If the dataset is partitioned, you can choose to:

* Train the model on the training set and test it on the test documents
* Train the model on the training set and update it with the test set
* Train the model with the whole dataset, regardless of any partition.

Evaluate a model
----------------

To evaluate a model, choose a metric and use the score() method of the metric class.

``` python
from evaluation_metrics.diversity_metrics import Topic_diversity

# Set metric parameters
td_parameters ={
'topk':10
}

metric = Topic_diversity(td_parameters) # Initialize metric
topic_diversity_score = metric.score(model_output) # Compute score of the metric
``` 

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

result.plot_all()
```

 
The result will provide best-seen value of the metric with the corresponding hyperparameter configuration, and the hyperparameters and metric value for each iteration of the optimization. To visualize this information, you can use the plot and plot_all methods of the result.

Available Models
----------------

* LDA
* LSI
* NMF
* HDP

Available Datasets
----------------

* 20Newsgroup
* Wikipedia

### Disclaimer

Similarly to [ `TensorFlow Datasets` ](https://github.com/tensorflow/datasets) and HuggingFace's [ `nlp` ](https://github.com/huggingface/nlp) library, we just downloaded and prepared public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license and to cite the right owner of the dataset.

If you're a dataset owner and wish to update any part of it, or do not want your dataset to be included in this library, please get in touch through a GitHub issue. 

If you're a dataset owner and wish to include your dataset in this library, please get in touch through a GitHub issue.  

Implement your own Model
------------------------

Models inherit from the class `Abstract_Model` defined in `models/model.py` .
To build your own model your class must override the `train_model(self, dataset, hyperparameters)` method which always require at least a `Dataset` object and a `Dictionary` of hyperparameters as input and should return a dictionary with the output of the model as output.

To better understand how a model work, let's have a look at the LDA implementation.
The first step in developing a custom model is to define the dictionary of default hyperparameters values:

``` python

hyperparameters = {
    'corpus': None,
    'num_topics': 100,
    'id2word': None,
    'alpha': 'symmetric',
    'eta': None,
    # ...
    'callbacks': None}

```

Defining the default hyperparameters values allows users to work on a subset of them without having to assign a value to each parameter.

The following step is the `train_model()` override:

``` python

 def train_model(self, dataset, hyperparameters={}, topics=10,
                    topic_word_matrix=True, topic_document_matrix=True):

```

The LDA method require a dataset, the hyperparameters dictionary and 3 extra (optional) arguments used to enable or disable the computing of outputs to enhance performances during optimization processes.

With the hyperparameters defaults, the ones in input and the dataset you should be able to write your own code and return as output a dictionary with at least 3 entries:

* `topics`: the list of the most significative words foreach topic (list of lists of strings).
* `topic-word-matrix`: an NxV matrix of weights where N is the number of topics and V is the vocabulary length.
* `topic-document-matrix`: an NxD matrix of weights where N is the number of topics and D is the number of documents in the corpus.

if your model support the training/test partitioning it should also return:

* `test-document-topic-matrix`: the document topic matrix of the test set.

In case the model isn't updated with the test set.
Or:

* `test-topics`: the list of the most significative words foreach topic (list of lists of strings) of the model updated with the test set.
* `test-topic-word-matrix`: an NxV matrix of weights where N is the number of topics and V is the vocabulary length of the model updated with the test set.
* `test-topic-document-matrix`: an NxD matrix of weights where N is the number of topics and D is the number of documents in the corpus of the model updated with the test set.

If the model is updated with the test set.
