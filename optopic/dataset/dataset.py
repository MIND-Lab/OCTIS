import json
from pathlib import Path


class Dataset:
    """
    Dataset handles a dataset and offer methods to
    access, save and edit the dataset data
    """

    def __init__(self, corpus=[], vocabulary={}, metadata={},
                 labels=[], edges=[]):
        """
        Initialize a dataset, parameters are optional
        if you want to load a dataset, initialize this
        class with default values and use the load method
        Parameters
        ----------
        corpus : corpus of the dataset
        vocabulary : vocabulary of the dataset
        metadata : metadata of the dataset
        labels : labels of the dataset
        edges : edges of the dataset
        """
        self.set_corpus(corpus)
        self.set_vocabulary(vocabulary)
        self.set_metadata(metadata)
        self.set_labels(labels)
        self.set_edges(edges)

    # Corpus setter

    def set_corpus(self, corpus):
        self.__corpus = corpus

    # Whole Corpus getter
    def get_corpus(self):
        if self.__corpus != []:
            return self.__corpus
        return False

    # Partitioned Corpus getter
    def get_partitioned_corpus(self, use_validation=True):
        last_training_doc = self.__metadata["last-training-doc"]
        #gestire l'eccezione se last_validation_doc non è definito, restituire
        # il validation vuoto
        if use_validation:
            last_validation_doc = self.__metadata["last-validation-doc"]
            if self.__corpus != [] and last_training_doc != 0:
                train_corpus = []
                test_corpus = []
                validation_corpus = []

                for i in range(last_training_doc):
                    train_corpus.append(self.__corpus[i])
                for i in range(last_training_doc, last_validation_doc):
                    validation_corpus.append(self.__corpus[i])
                for i in range(last_validation_doc, len(self.__corpus)):
                    test_corpus.append(self.__corpus[i])
                return train_corpus, validation_corpus, test_corpus
        else:
            if self.__corpus != [] and last_training_doc != 0:
                if "last-validation-doc" in self.__metadata.keys():
                    last_validation_doc = self.__metadata["last-validation-doc"]
                else:
                    last_validation_doc = 0

                train_corpus = []
                test_corpus = []
                for i in range(last_training_doc):
                    train_corpus.append(self.__corpus[i])

                if last_validation_doc != 0:
                    for i in range(last_validation_doc, len(self.__corpus)):
                        test_corpus.append(self.__corpus[i])
                else:
                    for i in range(last_training_doc, len(self.__corpus)):
                        test_corpus.append(self.__corpus[i])
                return train_corpus, test_corpus


    # Edges setter

    def set_edges(self, edges):
        self.__edges = edges

    # Edges getter
    def get_edges(self):
        if self.__edges != []:
            return self.__edges
        return False

    # Labels setter

    def set_labels(self, labels):
        self.__labels = labels

    # Labels getter
    def get_labels(self):
        if self.__labels != []:
            return self.__labels
        return False

    # Metadata setter

    def set_metadata(self, metadata):
        self.__metadata = metadata

    # Metadata getter
    def get_metadata(self):
        if self.__metadata != {}:
            return self.__metadata
        return False

    # Info getter
    def get_info(self):
        if self.__metadata != {}:
            if "info" in self.__metadata:
                return self.__metadata["info"]
        return False

    # Vocabulary setter

    def set_vocabulary(self, vocabulary):
        self.__vocabulary = vocabulary

    # Vocabulary getter
    def get_vocabulary(self):
        if self.__vocabulary != {}:
            return self.__vocabulary
        return False

    def _save_metadata(self, file_name):
        """
        Saves metadata in json serialized format
        Parameters
        ----------
        file_name : name of the file to write
        Returns
        -------
        True if the data is saved, False otherwise
        """
        data = self.get_metadata()
        if data:
            with open(file_name, 'w') as outfile:
                json.dump(data, outfile)
                return True
        return False

    def _load_metadata(self, file_name):
        """
        Loads metadata from json serialized format
        Parameters
        ----------
        file_name : name of the file to read
        Returns
        -------
        True if the data is updated, False otherwise
        """
        metadata = {}
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as metadata_file:
                metadata = json.load(metadata_file)
            self.set_metadata(metadata)
            return True
        return False

    def _save_corpus(self, file_name):
        """
        Saves corpus in a file, a line for each document
        Parameters
        ----------
        file_name : name of the file to write
        Returns
        -------
        True if the data is saved, False otherwise
        """
        data = self.get_corpus()
        if data:
            with open(file_name, 'w') as outfile:
                for element in data:
                    outfile.write("%s\n" % " ".join(element))
                return True
        return False

    def _load_corpus(self, file_name):
        """
        Loads corpus from a file
        Parameters
        ----------
        file_name : name of the file to read
        Returns
        -------
        True if the data is updated, False otherwise
        """
        corpus = []
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as corpus_file:
                for line in corpus_file:
                    corpus.append(line.split())
            self.set_corpus(corpus)
            return True
        return False

    def _save_edges(self, file_name):
        """
        Saves edges in a file, a line for each document
        Parameters
        ----------
        file_name : name of the file to write
        Returns
        -------
        True if the data is saved, False otherwise
        """
        data = self.get_edges()
        if data:
            with open(file_name, 'w') as outfile:
                for element in data:
                    outfile.write("%s\n" % element)
                return True
        return False

    def _load_edges(self, file_name):
        """
        Loads edges from a file
        Parameters
        ----------
        file_name : name of the file to read
        Returns
        -------
        True if the data is updated, False otherwise
        """
        edges = []
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as edges_file:
                for line in edges_file:
                    edges.append(line[0:len(line)-1])
            self.set_edges(edges)
            return True
        return False

    def _save_labels(self, file_name):
        """
        Saves the labels in a file, each line contains
        the labels of a single document
        Parameters
        ----------
        file_name : name of the file to write
        Returns
        -------
        True if the data is saved, False otherwise
        """
        data = self.get_labels()
        if data:
            with open(file_name, 'w') as outfile:
                for element in data:
                    outfile.write("%s\n" % json.dumps(element))
                return True
        return False

    def _load_labels(self, file_name):
        """
        Loads labels from a file
        Parameters
        ----------
        file_name : name of the file to read
        Returns
        -------
        True if the data is updated, False otherwise
        """
        labels = []
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as labels_file:
                for line in labels_file:
                    labels.append(json.loads(line))
            self.set_labels(labels)
            return True
        return False

    def _save_vocabulary(self, file_name):
        """
        Saves vocabulary dictionary in a file
        Parameters
        ----------
        file_name : name of the file to write
        Returns
        -------
        True if the data is saved, False otherwise
        """
        data = self.get_vocabulary()
        if data:
            with open(file_name, 'w') as outfile:
                for word, freq in data.items():
                    line = word+" "+str(freq)
                    outfile.write("%s\n" % line)
                return True
        return False

    def _load_vocabulary(self, file_name):
        """
        Loads vocabulary from a file
        Parameters
        ----------
        file_name : name of the file to read
        Returns
        -------
        True if the data is updated, False otherwise
        """
        vocabulary = {}
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as vocabulary_file:
                for line in vocabulary_file:
                    tmp = line.split()
                    vocabulary[tmp[0]] = float(tmp[1])
            self.set_vocabulary(vocabulary)
            return True
        return False

    def save(self, path):
        """
        Saves all the dataset info in a folder
        Parameters
        ----------
        path : path to the folder in wich files are saved.
               If the folder doesn't exist it will be created
        Returns
        -------
        True if the data is saved, False otherwise
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        corpus_saved = self._save_corpus(path+"/corpus.txt")
        vocabulary_saved = self._save_vocabulary(path+"/vocabulary.txt")
        self._save_labels(path+"/labels.txt")
        self._save_edges(path+"/edges.txt")
        metadata_saved = self._save_metadata(path+"/metadata.json")
        return corpus_saved and vocabulary_saved and metadata_saved

    def load(self, path):
        """
        Loads all the dataset from a folder
        Parameters
        ----------
        path : path of the folder to read
        Returns
        -------
        True if the data is saved, False otherwise
        """
        self.path=path
        
        
        corpus_readed = self._load_corpus(path+"/corpus.txt")
        vocabulary_readed = self._load_vocabulary(path+"/vocabulary.txt")
        self._load_labels(path+"/labels.txt")
        self._load_edges(path+"/edges.txt")
        metadata_readed = self._load_metadata(path+"/metadata.json")
        return corpus_readed and vocabulary_readed and metadata_readed