from models.model import Abstract_Model
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import DictionaryLearning
from gensim.models import nmf
import gensim.corpora as corpora
import configuration.citations as citations
import configuration.defaults as defaults
import scipy.sparse

class NMF_gensim(Abstract_Model):

    id2word = None
    id_corpus = None
    use_partitions = True
    update_with_test = False

    def info(self):
        """
        Returns model informations
        """
        return {
            "citation": citations.models_NMF,
            "name": "NMF, Non-negative Matrix Factorization"
        }

    def hyperparameters_info(self):
        """
        Returns hyperparameters informations
        """
        return defaults.NMF_gensim_hyperparameters_info

    def partitioning(self, use_partitions, update_with_test=False):
        """
        Handle the partitioning system to use and reset the model to perform
        new evaluations

        Parameters
        ----------
        use_partitions: True if train/set partitioning is needed, False
                        otherwise
        update_with_test: True if the model should be updated with the test set,
                          False otherwise
        """
        self.use_partitions = use_partitions
        self.update_with_test = update_with_test
        self.id2word = None
        self.id_corpus = None

    def train_model(self, dataset, hyperparameters={}, top_words=10,
                    topic_word_matrix=True, topic_document_matrix=True):
        """
        Train the model and return output

        Parameters
        ----------
        dataset : dataset to use to build the model
        hyperparameters : hyperparameters to build the model
        top_words : if greather than 0 returns the most significant words
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
        partition = []
        if self.use_partitions:
            partition = dataset.get_partitioned_corpus()
        else:
            partition = [dataset.get_corpus(), []]

        if self.id2word == None:
            self.id2word = corpora.Dictionary(dataset.get_corpus())

        if self.id_corpus == None:
            self.id_corpus = [self.id2word.doc2bow(
                document) for document in partition[0]]

        hyperparameters["corpus"] = self.id_corpus
        hyperparameters["id2word"] = self.id2word
        self.hyperparameters.update(hyperparameters)

        self.trained_model = nmf.Nmf(**self.hyperparameters)

        result = {}

        if topic_word_matrix:
            result["topic-word-matrix"] = self.trained_model.get_topics()

        if top_words > 0:
            topics_output = []
            for topic in result["topic-word-matrix"]:
                top_k = np.argsort(topic)[-top_words:]
                top_k_words = list(reversed([self.id2word[i] for i in top_k]))
                topics_output.append(top_k_words)
            result["topics"] = topics_output

        if topic_document_matrix:
            result["topic-document-matrix"] = self._get_topic_document_matrix()

        if self.use_partitions:
            new_corpus = [self.id2word.doc2bow(
                document) for document in partition[1]]
            if self.update_with_test:
                self.trained_model.update(new_corpus)
                self.id_corpus.extend(new_corpus)

                if topic_word_matrix:
                    result["test-topic-word-matrix"] = self.trained_model.get_topics()

                if top_words > 0:
                    topics_output = []
                    for topic in result["test-topic-word-matrix"]:
                        top_k = np.argsort(topic)[-top_words:]
                        top_k_words = list(
                            reversed([self.id2word[i] for i in top_k]))
                        topics_output.append(top_k_words)
                    result["test-topics"] = topics_output

                if topic_document_matrix:
                    result["test-topic-document-matrix"] = self._get_topic_document_matrix()

            else:
                test_document_topic_matrix = []
                for document in new_corpus:
                    test_document_topic_matrix.append(
                        self.trained_model[document])
                result["test-document-topic-matrix"] = np.array(
                    test_document_topic_matrix)

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
        for document in self.id_corpus:
            doc_topic_tuples.append(
                self.trained_model.get_document_topics(document,
                                                       minimum_probability=0))

        topic_document = np.zeros((
            self.hyperparameters["num_topics"],
            len(doc_topic_tuples)))

        for ndoc in range(len(doc_topic_tuples)):
            document = doc_topic_tuples[ndoc]
            for topic_tuple in document:
                topic_document[topic_tuple[0]][ndoc] = topic_tuple[1]
        return topic_document


class NMF_scikit(Abstract_Model):

    id2word = None
    id_corpus = None
    use_partitions = True
    update_with_test = False
    hyperparameters = {"num_topics": 100,
                       "init": "random", "alpha": 0, "l1_ratio": 0}

    def hyperparameters_info(self):
        """
        Returns hyperparameters informations
        """
        return defaults.NMF_scikit_hyperparameters_info


    def partitioning(self, use_partitions, update_with_test=False):
        """
        Handle the partitioning system to use and reset the model to perform
        new evaluations

        Parameters
        ----------
        use_partitions: True if train/set partitioning is needed, False
                        otherwise
        update_with_test: True if the model should be updated with the test set,
                          False otherwise
        """
        self.use_partitions = use_partitions
        self.update_with_test = update_with_test
        self.id2word = None
        self.id_corpus = None

    def train_model(self, dataset, hyperparameters={}, topics=10,
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
        if self.id2word == None or self.id_corpus == None:
            vectorizer = TfidfVectorizer(min_df=0.0)
            corpus = dataset.get_corpus()
            real_corpus = []
            for document in corpus:
                real_corpus.append(" ".join(document))
            X = vectorizer.fit_transform(real_corpus)

            lista = vectorizer.get_feature_names()
            self.id2word = {i: lista[i] for i in range(0, len(lista))}
            if self.use_partitions:
                ltd = dataset.get_metadata()[
                    "last-training-doc"]
                self.id_corpus = X[0:ltd]
                self.new_corpus = X[ltd:]
            else:
                self.id_corpus = X

        hyperparameters["corpus"] = self.id_corpus
        hyperparameters["id2word"] = self.id2word
        self.hyperparameters.update(hyperparameters)

        model = NMF(
            n_components=self.hyperparameters["num_topics"],
            init=self.hyperparameters["init"],
            alpha=self.hyperparameters["alpha"],
            l1_ratio=self.hyperparameters["l1_ratio"])

        W = model.fit_transform(self.id_corpus)
        #W = W / W.sum(axis=1, keepdims=True)
        H = model.components_
        #H = H / H.sum(axis=1, keepdims=True)

        result = {}

        if topic_word_matrix:
            result["topic-word-matrix"] = H

        if topics > 0:
            result["topics"] = self.get_topics(H, topics)

        if topic_document_matrix:
            result["topic-document-matrix"] = np.array(W).transpose()

        if self.use_partitions:
            if self.update_with_test:
               # NOT IMPLEMENTED YET

                if topic_word_matrix:
                    result["test-topic-word-matrix"] = W

                if topics > 0:
                    result["test-topics"] = self.get_topics(W, topics)

                if topic_document_matrix:
                    result["test-topic-document-matrix"] = H

            else:
                result["test-document-topic-matrix"] = model.transform(
                    self.new_corpus)

        return result

    def get_topics(self, H, topics):
        topic_list = []
        for topic in H:
            words_list = sorted(
                list(enumerate(topic)), key=lambda x: x[1])
            topk = [tup[0] for tup in words_list[0:topics]]
            topic_list.append([self.id2word[i] for i in topk])
        return topic_list


def NMF_Model(implementation="gensim"):
    """
    Choose NMF implementation and return the correct model

    Parameters
    ----------
    implementation: implementation of NMF to use
                    available 'gensim' and 'scikit'
                    default 'gensim'

    Returns
    -------
    model: an initialized model of the choosen implementation
    """
    if implementation == "gensim":
        return NMF_gensim()
    if implementation == "scikit":
        return NMF_scikit()
