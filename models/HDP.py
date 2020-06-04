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
    dataset = None

    def info(self):
        return {
            "citation": r"""
@inproceedings{DBLP:conf/nips/TehJBB04,
  author    = {Yee Whye Teh and
               Michael I. Jordan and
               Matthew J. Beal and
               David M. Blei},
  title     = {Sharing Clusters among Related Groups: Hierarchical Dirichlet Processes},
  booktitle = {Advances in Neural Information Processing Systems 17 [Neural Information
               Processing Systems, {NIPS} 2004, December 13-18, 2004, Vancouver,
               British Columbia, Canada]},
  pages     = {1385--1392},
  year      = {2004},
  url       = {http://papers.nips.cc/paper/2698-sharing-clusters-among-related-groups-hierarchical-dirichlet-processes},
  timestamp = {Fri, 06 Mar 2020 16:59:17 +0100},
  biburl    = {https://dblp.org/rec/conf/nips/TehJBB04.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
            """,
            "name": "HDP, Hierarchical Dirichlet Process"
        }

    def train_model(self, dataset, hyperparameters={}, topics=10,
                    topic_word_matrix=True, topic_document_matrix=True,
                    use_partitions=True, update_with_test=False):
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
        partition = []
        if use_partitions:
            partition = dataset.get_partitioned_corpus()

        if self.id2word == None:
            self.id2word = corpora.Dictionary(dataset.get_corpus())

        if self.id_corpus == None:
            self.id_corpus = [self.id2word.doc2bow(
                document) for document in dataset.get_corpus()]

        if self.dataset == None:
            self.dataset = dataset

        self.hyperparameters.update(hyperparameters)
        hyperparameters = self.hyperparameters

        self.trained_model = hdpmodel.HdpModel(
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
            result["topic-word-matrix"] = self.trained_model.get_topics()

        if topics > 0:
            result["topics"] = self._get_topics_words(topics)

        if topic_document_matrix:
            result["topic-document-matrix"] = self._get_topic_document_matrix()

        if use_partitions:
            new_corpus = [self.id2word.doc2bow(
                document) for document in partition[1]]
            if update_with_test:
                self.trained_model.update(new_corpus)

                if topic_word_matrix:
                    result["test-topic-word-matrix"] = self.trained_model.get_topics()

                if topics > 0:
                    result["test-topics"] = self._get_topics_words(topics)

                if topic_document_matrix:
                    result["test-topic-document-matrix"] = self._get_topic_document_matrix()

            else:
                test_document_topic_matrix = []
                for document in new_corpus:
                    test_document_topic_matrix.append(
                        self.trained_model[document])
                result["test-document-topic-matrix"] = test_document_topic_matrix

        return result

    def _get_topics_words(self, topics):
        """
        Return the most significative words for each topic.
        """
        if topics > 0:
            topic_terms = []
        for i in range(len(self.trained_model.get_topics())):
            topic_terms.append(self.trained_model.show_topic(
                i,
                topics,
                False,
                True
            ))
        return topic_terms

    def _get_topic_document_matrix(self):
        """
        Return the topic representation of the
        corpus
        """
        doc_topic_tuples = []
        for document in self.dataset.get_corpus():
            doc_topic_tuples.append(self.trained_model[
                self.id2word.doc2bow(document)])

        topic_document = np.zeros((
            len(self.trained_model.get_topics()),
            len(doc_topic_tuples)))

        for ndoc in range(len(doc_topic_tuples)):
            document = doc_topic_tuples[ndoc]
            for topic_tuple in document:
                topic_document[topic_tuple[0]][ndoc] = topic_tuple[1]
        return topic_document
