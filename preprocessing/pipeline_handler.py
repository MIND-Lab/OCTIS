import preprocessing.tools as tools
import spacy
from nltk.corpus import stopwords


class Pipeline_handler:
    """
    Pipeline_handler is a class used to describe a
    preprocessing pipeline and execute it with the
    'process' method
    """

    nlp = spacy.load("en")
    stop_words = stopwords.words('english')

    min_words_for_doc = 0
    words_min_freq = 0
    words_max_freq = 0
    num_proc = 1

    stop_words_extension = []
    pos = ['NOUN', 'ADJ', 'VERB', 'ADV']

    remove_punctuation = True
    remove_stopwords = True
    lemmatize = True
    filter_words = True
    multiprocess = False
    display_progress = False

    def __init__(self, dataset):
        """
        Initialize a Pipeline_handler

        Parameters
        ----------
        dataset : dictionary with corpus, labels and other data
                  about the dataset
        """
        self.words_max_freq = len(dataset["corpus"])
        self.dataset = dataset

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
        self.stop_words.extend(self.stop_words_extension)
        pipeline = []

        if self.display_progress:
            print("Initializing preprocessing")

        # Add each enabled step to the pipeline
        if self.remove_stopwords:
            pipeline.append(tools.remove_stopwords)
        if self.lemmatize:
            pipeline.append(tools.lemmatization)
        if self.filter_words:
            pipeline.append(tools.filter_words)

        corpus = self.dataset["corpus"]
        categories = []
        if "doc_labels" in self.dataset:
            categories = self.dataset["doc_labels"]

        if self.multiprocess:
            pool = tools.create_pool(self.num_proc)

        if self.remove_punctuation:
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

            arguments = []
            if step.__name__ == "filter_words":
                arguments = tools.words_to_remove(
                    corpus,
                    self.words_min_freq,
                    self.words_max_freq)
            elif step.__name__ == "lemmatization":
                arguments = [self.nlp, self.pos]
            elif step.__name__ == "remove_stopwords":
                arguments = self.stop_words
            else:
                arguments = []

            if self.multiprocess:
                corpus = tools._multiprocess(
                    pool,
                    self.num_proc,
                    step,
                    corpus,
                    arguments)
            else:
                corpus = step(corpus, arguments)

            if self.display_progress:
                print("  step: "+step.__name__+" executed")

        result = tools.remove_docs(
            corpus,
            self.min_words_for_doc,
            categories)

        if self.multiprocess:
            pool.close()

        print("Preprocess Done!")
        return result
