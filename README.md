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

```python
import preprocessing.sources.newsgroup as source
dataset = source.retrieve()
```

Or use your own.

```python
import preprocessing.sources.custom_dataset as source
dataset = source.retrieve("path\to\dataset")
```

A custom dataset must have a document for each line of the file.

Preprocess
----------

To preprocess a dataset Initialize a Pipeline_handler and use the preprocess method.

```python
from preprocessing.pipeline_handler import Pipeline_handler

pipeline_handler = Pipeline_handler(dataset) # Initialize pipeline handler
preprocessed = pipeline_handler.preprocess() # preprocess

preprocessed.save("dataset_folder") # Save the preprocessed dataset
```

For the customization of the preprocess pipeline see the optimization demo example in the examples folder.

Build a model
-------------

To build a model, load a preprocessed dataset, customize the model hyperparameters and use the train_model() method of the model class.

```python
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

Evaluate a model
----------------

To evaluate a model, choose a metric and use the score() method of the metric class.

```python
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

To optimize a model you need to select a model, a dataset, a metric and the search space of the hyperparameters to optimize.

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
