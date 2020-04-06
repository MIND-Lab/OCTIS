from models.LSI import LSI_Model
from dataset.dataset import Dataset
import json


dataset = Dataset()
dataset.load("preprocessed_datasets/newsgroup/newsgroup_lemmatized_5")

hyperparameters = {}
hyperparameters["num_topics"] = 10

model = LSI_Model(dataset, hyperparameters)  # Create model

model.train_model()  # Train the model

model.make_topic_word_matrix()  # compute topic word matrix
topic_word_matrix = model.topic_word_matrix

print(topic_word_matrix)
