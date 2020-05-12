from models.model import Abstract_Model
import numpy as np
from gensim.models import hdpmodel
import gensim.corpora as corpora


class HDP_Model(Abstract_Model):

    hyperparameters = {
        'corpus': None,
        'id2word': None,
        'max_chunks': None,
        'max_time': None,
        'chunksize': 256,
        'kappa': 1.0,
        'tau': 64.0,
        'K': 15,
        'T': 150,
        'alpha': 1,
        'gamma': 1,
        'eta': 0.01,
        'scale': 1.0,
        'var_convergence': 0.0001,
        'outputdir': None,
        'random_state': None}

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
        topics : if greather than 0 returns the top k most significant
                 words for each topic in the output
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

        trained_model = hdpmodel.HdpModel(
            corpus=self.id_corpus,
            id2word=self.id2word,
            max_chunks=hyperparameters["max_chunks"],
            max_time=hyperparameters["max_time"],
            chunksize=hyperparameters["chunksize"],
            kappa=hyperparameters["kappa"],
            tau=hyperparameters["tau"],
            K=hyperparameters["K"],
            T=hyperparameters["T"],
            alpha=hyperparameters["alpha"],
            gamma=hyperparameters["gamma"],
            eta=hyperparameters["eta"],
            scale=hyperparameters["scale"],
            var_converge=hyperparameters["var_convergence"],
            outputdir=hyperparameters["outputdir"],
            random_state=hyperparameters["random_state"]
        )

        result = {}

        if topic_word_matrix:
            result["topic-word-matrix"] = trained_model.get_topics()

        if topics > 0:
            topic_terms = []
        for i in range(len(trained_model.get_topics())):
            topic_terms.append(trained_model.show_topic(
                i,
                topics,
                False,
                True
            ))

            result["topics"] = topic_terms

        if topic_document_matrix:
            doc_topic_tuples = []
            for document in dataset.get_corpus():
                doc_topic_tuples.append(trained_model[
                    self.id2word.doc2bow(document)])

            topic_document = np.zeros((
                len(trained_model.get_topics()),
                len(doc_topic_tuples)))

            for ndoc in range(len(doc_topic_tuples)):
                document = doc_topic_tuples[ndoc]
                for topic_tuple in document:
                    topic_document[topic_tuple[0]][ndoc] = topic_tuple[1]

            result["topic-document-matrix"] = topic_document

        return result
