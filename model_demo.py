from models.LDA import LDA_Model
from dataset.dataset import Dataset
import json

dataset = Dataset()
dataset.load("preprocessed_datasets/newsgroup/newsgroup_lemmatized_5")

hyperparameters = {}
hyperparameters["num_topics"] = 10
hyperparameters["alpha"] = 'auto'
hyperparameters["eta"] = None

model = LDA_Model(dataset, hyperparameters)
model.build_model()

topic_word_matrix = model.topic_word_matrix

print(topic_word_matrix)
