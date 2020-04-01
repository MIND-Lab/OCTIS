from models.interface import Abstract_Model
from dataset.dataset import Dataset

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

class LDA_Model(Abstract_Model):
    
    def model_builder(self, data, hyperparameters):
        id2word = corpora.Dictionary(data.get_corpus())
        corpus = data.get_corpus()
        id_corpus = [id2word.doc2bow(document) for document in corpus]
        model = gensim.models.ldamodel.LdaModel(
          corpus= id_corpus,
          num_topics= hyperparameters["num_topics"],
          id2word= id2word,
          alpha = hyperparameters["alpha"],
          eta = hyperparameters["eta"])
        
        return model.print_topics()

