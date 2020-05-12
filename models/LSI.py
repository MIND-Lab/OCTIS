from models.model import Abstract_Model
from gensim.models import lsimodel
import numpy as np
import gensim.corpora as corpora


class LSI_Model(Abstract_Model):

    hyperparameters = {
        'corpus': None,
        'num_topics': 100,
        'id2word': None,
        'distributed': False,
        'chunksize': 20000,
        'decay': 1.0,
        'onepass': True,
        'power_iters': 2,
        'extra_samples': 100}

    id2word = None
    id_corpus = None

    def train_model(self, dataset, hyperparameters, topics=10,
                    topic_word_matrix=True, topic_document_matrix=True):
        """
        Train the model and return output

        Parameters
        ----------
        dataset : dataset to use to build the model
        hyperparameters : hyperparameters to build the model
        topics : if greather than 0returns the most significant words
                 for each topic in the output
                 Default True
        topic_word_matrix : if True returns the topic word matrix in the output
                            Default True
        topic_document_matrix : if True returns the topic document
                                matrix in the output
                                Default True

        Returns
        -------
        result : dictionary with up to 3 entries,
                 'topics', 'topic-word-matrix' and 
                 'topic-document-matrix'
        """
        if self.id2word == None:
            self.id2word = corpora.Dictionary(dataset.get_corpus())
        if self.id_corpus == None:
            self.id_corpus = [self.id2word.doc2bow(
                document) for document in dataset.get_corpus()]

        self.hyperparameters.update(hyperparameters)
        hyperparameters = self.hyperparameters

        hyperparameters = self.hyperparameters
        trained_model = lsimodel.LsiModel(
            corpus=self.id_corpus,
            id2word=self.id2word,
            num_topics=hyperparameters["num_topics"],
            distributed=hyperparameters["distributed"],
            chunksize=hyperparameters["chunksize"],
            decay=hyperparameters["decay"],
            onepass=hyperparameters["onepass"],
            power_iters=hyperparameters["power_iters"],
            extra_samples=hyperparameters["extra_samples"])

        result = {}

        if topic_word_matrix:

            topic_word_matrix = trained_model.get_topics()
            normalized = []
            for words_w in topic_word_matrix:
                minimum = min(words_w)
                words = words_w - minimum
                normalized.append([float(i)/sum(words) for i in words])
            topic_word_matrix = np.array(normalized)

            result["topic-word-matrix"] = topic_word_matrix

        if topics > 0:
            topic_terms = []
            for i in range(self.hyperparameters["num_topics"]):
                topic_words_list = []
                for word_tuple in trained_model.show_topic(i):
                    topic_words_list.append(word_tuple[0])
                topic_terms.append(topic_words_list)

            result["topics"] = topic_terms

        if topic_document_matrix:
            bow = [self.id2word.doc2bow(document)
                   for document in dataset.get_corpus()]
            topic_weights = trained_model[bow]

            topic_document = []

            for document_topic_weights in topic_weights:

                # Find min e max topic_weights values
                minimum = document_topic_weights[0][1]
                maximum = document_topic_weights[0][1]
                for topic in document_topic_weights:
                    if topic[1] > maximum:
                        maximum = topic[1]
                    if topic[1] < minimum:
                        minimum = topic[1]

                # For each topic compute normalized weight
                # in the form (value-min)/(max-min)
                topic_w = []
                for topic in document_topic_weights:
                    topic_w.append((topic[1]-minimum)/(maximum-minimum))
                topic_document.append(topic_w)

            result["topic-document-matrix"] = np.array(
                topic_document).transpose()

        return result
