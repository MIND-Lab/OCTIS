from models.LDA import LDA_Model
from dataset.dataset import Dataset

dataset = Dataset()
dataset.load("preprocessed_datasets/dblp/dblp_0")

hyperparameters = {}
hyperparameters["num_topics"] = 10
hyperparameters["alpha"] = 'auto'
hyperparameters["eta"] = None

c = LDA_Model(dataset, hyperparameters)
c.build_model()
print(c.topic_word_matrix)