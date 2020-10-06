import preprocessing.tools as tools
import spacy
from nltk.corpus import stopwords as nltk_stopwords


class Pipeline_handler:
    """
    Pipeline_handler is a class used to describe a
    preprocessing pipeline and execute it with the
    'process' method
    """

    def __init__(self, dataset, min_words_for_doc=0, words_min_freq = 0, words_max_freq = 1,
                 num_processes=1, stopwords='english', pos=['NOUN', 'ADJ', 'VERB', 'ADV'],
                 remove_punctuation=True, remove_stopwords=True, lemmatize=True,
                 filter_words=True, display_progress=False):
        """
        Initialize a Pipeline_handler

        Parameters
        ----------
        dataset : dictionary with corpus, labels and other data
                  about the dataset
        """
        self.words_max_freq = len(dataset["corpus"])
        self.dataset = dataset

        self.min_words_for_doc = min_words_for_doc
        self.words_min_freq = words_min_freq
        self.words_max_freq = words_max_freq
        self.num_proc = num_processes
        if self.num_proc > 1:
            self.multiprocess = True

        if type(stopwords) == str:
            self.stop_words = nltk_stopwords.words(stopwords)
        elif type(stopwords) == list:
            self.stop_words = stopwords
        else:
            print("type of stopwords parameter must be str or list of str")
            self.stop_words = []

        self.pos = pos
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.filter_words = filter_words
        self.display_progress = display_progress

    def preprocess(self):
        """
        Preprocess the dataset

        Parameters
        ----------
        preprocess has no parameters.
        The steps are enabled/disabled by setting
        the relative variables True/False
        Parameters are customizable by setting
        the relative variables

        Returns
        -------
        Preprocessed dataset object
        """
        #self.stop_words.extend(self.stop_words_extension)
        pipeline = []
        extra_data = "Steps:\n"
        parameters = "Parameters:\n"

        if self.display_progress:
            print("Initializing preprocessing")

        # Add each enabled step to the pipeline
        if self.lemmatize:
            pipeline.append(tools.lemmatization)
        if self.remove_stopwords:
            pipeline.append(tools.remove_stopwords)
            # parameters += "  stopwords extension:" + str(self.stop_words_extension) + "\n"
        if self.filter_words:
            pipeline.append(tools.filter_words)
            parameters += "  removed words with less than " + str(self.words_min_freq) +\
                          " or more than " + str(self.words_max_freq) + \
                          " documents with an occurrence of the word in corpus\n"
            
            parameters += "  removed documents with less than " + str(self.min_words_for_doc) + " words"

        corpus = self.dataset["corpus"]
        categories = []
        if "doc_labels" in self.dataset:
            categories = self.dataset["doc_labels"]

        partition = 0
        if "partition" in self.dataset:
            partition = self.dataset["partition"]
        edges = []
        if "edges" in self.dataset:
            edges = self.dataset["edges"]
        info = {}
        if "info" in self.dataset:
            info = self.dataset["info"]

        if self.multiprocess:
            pool = tools.create_pool(self.num_proc)

        if self.remove_punctuation:
            extra_data += "  remove_punctuation\n"
            if self.display_progress:
                print("  step: remove_punctuation")

            if self.multiprocess:
                corpus = tools._multiprocess(
                    pool,
                    self.num_proc,
                    tools.remove_punctuation,
                    corpus)
            else:
                corpus = tools.remove_punctuation(corpus)

            if self.display_progress:
                print("  step: remove_punctuation executed")

        corpus = list(tools.create_bags_of_words(corpus))

        # Execute each step in the pipeline
        for step in pipeline:

            if self.display_progress:
                print("  step: "+step.__name__)

            if step.__name__ == "filter_words":
                arguments = tools.words_to_remove(
                    corpus,
                    self.words_min_freq,
                    self.words_max_freq)
            elif step.__name__ == "lemmatization":
                arguments = self.pos
            elif step.__name__ == "remove_stopwords":
                arguments = self.stop_words
            else:
                arguments = []

            if self.multiprocess:
                corpus = tools._multiprocess(pool, self.num_proc, step, corpus, arguments)
            else:
                corpus = step(corpus, arguments)

            extra_data += "  "+step.__name__+"\n"

            if self.display_progress:
                print("  step: "+step.__name__+" executed")
        
        extra_data += "  remove_docs\n"
        extra_data += parameters

        result = tools.remove_docs(corpus, self.min_words_for_doc, categories,
                                   partition, edges, extra_data, info)

        if self.multiprocess:
            pool.close()

        print("Preprocess Done!")
        return result
