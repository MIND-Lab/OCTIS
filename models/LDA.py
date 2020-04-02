from models.interface import Abstract_Model
from dataset.dataset import Dataset

import re

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

class LDA_Model(Abstract_Model):
    


    def build_model(self):
        id2word = corpora.Dictionary(self.dataset.get_corpus())
        corpus = self.dataset.get_corpus()
        num_topics = self.hyperparameters["num_topics"]
        id_corpus = [id2word.doc2bow(document) for document in corpus]
        model = gensim.models.ldamodel.LdaModel(
          corpus= id_corpus,
          num_topics= num_topics,
          id2word= id2word,
          alpha = self.hyperparameters["alpha"],
          eta = self.hyperparameters["eta"])
        metadata = self.dataset.get_metadata()

        topic_word_matrix = []
        vocabulary_length = metadata["vocabulary_length"]

        topic_word_tuples = model.print_topics(num_words=vocabulary_length)
        for i in range(num_topics):
          topic_word_matrix.append([0.0] * vocabulary_length)
          topic = topic_word_tuples[i] 
          words_weight_list = topic[1]
          tmp = words_weight_list.split("+")
          for el in tmp:
            weight_word = el.split("*")
            weight = float(weight_word[0])
            word = re.sub("[^a-zA-Z]+", "", weight_word[1])
            topic_word_matrix[i][self.word_id[word]] = weight            
            
        self.topic_word_matrix = topic_word_matrix
        
        self.topic_representation = model.get_document_topics(id_corpus)

