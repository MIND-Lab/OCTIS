from models.model import Abstract_Model
import numpy as np
from gensim.models import nmf
import gensim.corpora as corpora


class NMF_Model(Abstract_Model):

    hyperparameters = {
        'corpus': None,
        'num_topics': 100,
        'id2word': None,
        'chunksize': 2000,
        'passes': 1,
        'kappa': 1.0,
        'minimum_probability': 0.01,
        'w_max_iter': 200,
        'w_stop_condition': 0.0001,
        'h_max_iter': 50,
        'h_stop_condition': 0.001,
        'eval_every': 10,
        'normalize': True,
        'random_state': None}

    id2word = None
    id_corpus = None
    dataset = None

    def train_model(self, dataset, hyperparameters, topics=10,
                    topic_word_matrix=True, topic_document_matrix=True):
        """
        Train the model and return output

        Parameters
        ----------
        dataset : dataset to use to build the model
        hyperparameters : hyperparameters to build the model
        topics : if greather than 0 returns the most significant words
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

        if self.dataset == None:
            self.dataset = dataset

        self.hyperparameters.update(hyperparameters)
        hyperparameters = self.hyperparameters

        self.trained_model = nmf.Nmf(
            corpus=self.id_corpus,
            id2word=self.id2word,
            num_topics=hyperparameters["num_topics"],
            chunksize=hyperparameters["chunksize"],
            passes=hyperparameters["passes"],
            kappa=hyperparameters["kappa"],
            minimum_probability=hyperparameters["minimum_probability"],
            w_max_iter=hyperparameters["w_max_iter"],
            w_stop_condition=hyperparameters["w_stop_condition"],
            h_max_iter=hyperparameters["h_max_iter"],
            h_stop_condition=hyperparameters["h_stop_condition"],
            eval_every=hyperparameters["eval_every"],
            normalize=hyperparameters["normalize"],
            random_state=hyperparameters["random_state"])

        result = {}

        if topic_word_matrix:
            result["topic-word-matrix"] = self.trained_model.get_topics()

        if topics > 0:
            result["topics"] = self._get_topics_words(topics)

        if topic_document_matrix:
            result["topic-document-matrix"] = self._get_topic_document_matrix()

        return result

    def _get_topics_words(self, topk):
        """
        Return the most significative words for each topic.
        """
        topic_terms = []
        for i in range(self.hyperparameters["num_topics"]):
            topic_words_list = []
            for word_tuple in self.trained_model.get_topic_terms(i, topk):
                topic_words_list.append(self.id2word[word_tuple[0]])
            topic_terms.append(topic_words_list)
        return topic_terms

    def _get_topic_document_matrix(self):
        """
        Return the topic representation of the
        corpus
        """
        doc_topic_tuples = []
        for document in self.dataset.get_corpus():
            doc_topic_tuples.append(self.trained_model.get_document_topics(
                self.id2word.doc2bow(document)))

        topic_document = np.zeros((
            self.hyperparameters["num_topics"],
            len(doc_topic_tuples)))

        for ndoc in range(len(doc_topic_tuples)):
            document = doc_topic_tuples[ndoc]
            for topic_tuple in document:
                topic_document[topic_tuple[0]][ndoc] = topic_tuple[1]
        return topic_document
