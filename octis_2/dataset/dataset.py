import json
from pathlib import Path


class Dataset:
    """
    Dataset handles a dataset and offer methods to
    access, save and edit the dataset data
    """

    def __init__(self, corpus=None, vocabulary=None, metadata=None,
                 labels=None, edges=None):
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
        self.__corpus = corpus
        self.__vocabulary = vocabulary
        self.__metadata = metadata
        self.__labels = labels
        self.__edges = edges

    def get_corpus(self):
        return self.__corpus

    # Partitioned Corpus getter
    def get_partitioned_corpus(self, use_validation=True):
        last_training_doc = self.__metadata["last-training-doc"]
        #gestire l'eccezione se last_validation_doc non è definito, restituire
        # il validation vuoto
        if use_validation:
            last_validation_doc = self.__metadata["last-validation-doc"]
            if self.__corpus is not None and last_training_doc != 0:
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
            if self.__corpus is not None and last_training_doc != 0:
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

    # Edges getter
    def get_edges(self):
        return self.__edges

    # Labels getter
    def get_labels(self):
        return self.__labels

    # Metadata getter
    def get_metadata(self):
        return self.__metadata

    # Info getter
    def get_info(self):
        if "info" in self.__metadata:
            return self.__metadata["info"]
        else:
            return None

    # Vocabulary getter
    def get_vocabulary(self):
        return self.__vocabulary

    def _save_metadata(self, file_name):
        """
        Saves metadata in json serialized format
        Parameters
        ----------
        file_name : name of the file to write
        Returns
        -------
        True if the data is saved
        """
        data = self.get_metadata()
        if data is not None:
            with open(file_name, 'w') as outfile:
                json.dump(data, outfile)
                return True
        else:
            raise Exception("error in saving metadata")

    def _load_metadata(self, file_name):
        """
        Loads metadata from json serialized format
        Parameters
        ----------
        file_name : name of the file to read
        """
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as metadata_file:
                metadata = json.load(metadata_file)
            self.__metadata = metadata

    def _save_corpus(self, file_name):
        """
        Saves corpus in a file, a line for each document
        Parameters
        ----------
        file_name : name of the file to write
        """
        data = self.get_corpus()
        if data is not None:
            with open(file_name, 'w') as outfile:
                for element in data:
                    outfile.write("%s\n" % " ".join(element))
        else:
            raise Exception("error in saving metadata")

    def _load_corpus(self, file_name):
        """
        Loads corpus from a file
        Parameters
        ----------
        file_name : name of the file to read
        """
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as corpus_file:
                corpus = [line.strip().split() for line in corpus_file]
            self.__corpus = corpus
        else:
            raise Exception("error in loading corpus")

    def _save_edges(self, file_name):
        """
        Saves edges in a file, a line for each document
        Parameters
        ----------
        file_name : name of the file to write
        """
        data = self.get_edges()
        if data is not None:
            with open(file_name, 'w') as outfile:
                for element in data:
                    outfile.write("%s\n" % element)
        else:
            raise Exception("error in saving edges")

    def _load_edges(self, file_name):
        """
        Loads edges from a file
        Parameters
        ----------
        file_name : name of the file to read
        """
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as edges_file:
                edges = [line[0:len(line)-1] for line in edges_file]
            self.__edges = edges

    def _save_labels(self, file_name):
        """
        Saves the labels in a file, each line contains
        the labels of a single document
        Parameters
        ----------
        file_name : name of the file to write
        """
        data = self.get_labels()
        if data is not None:
            with open(file_name, 'w') as outfile:
                for element in data:
                    outfile.write("%s\n" % json.dumps(element))
        else:
            raise Exception("error in saving labels")

    def _load_labels(self, file_name):
        """
        Loads labels from a file
        Parameters
        ----------
        file_name : name of the file to read
        ----------
        """
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as labels_file:
                labels = [json.loads(line.strip()) for line in labels_file]
            self.__labels = labels

    def _save_vocabulary(self, file_name):
        """
        Saves vocabulary dictionary in a file
        Parameters
        ----------
        file_name : name of the file to write
        -------
        """
        data = self.get_vocabulary()
        if data is not None:
            with open(file_name, 'w') as outfile:
                for word, freq in data.items():
                    line = word+" "+str(freq)
                    outfile.write("%s\n" % line)
        else:
            raise Exception("error in saving vocabulary")

    def _load_vocabulary(self, file_name):
        """
        Loads vocabulary from a file
        Parameters
        ----------
        file_name : name of the file to read
        """
        vocabulary = {}
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as vocabulary_file:
                for line in vocabulary_file:
                    tmp = line.split()
                    vocabulary[tmp[0]] = float(tmp[1])
            self.__vocabulary = vocabulary
        else:
            raise Exception("error in loading vocabulary")

    def save(self, path):
        """
        Saves all the dataset info in a folder
        Parameters
        ----------
        path : path to the folder in wich files are saved.
               If the folder doesn't exist it will be created
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        try:
            self._save_corpus(path+"/corpus.txt")
            self._save_vocabulary(path+"/vocabulary.txt")
            self._save_labels(path+"/labels.txt")
            self._save_edges(path+"/edges.txt")
            self._save_metadata(path+"/metadata.json")
        except:
            raise Exception("error in saving the dataset")

    def load(self, path):
        """
        Loads all the dataset from a folder
        Parameters
        ----------
        path : path of the folder to read
        """
        try:
            self.path=path
            self._load_corpus(path+"/corpus.txt")
            self._load_vocabulary(path+"/vocabulary.txt")
            self._load_labels(path+"/labels.txt")
            self._load_edges(path+"/edges.txt")
            self._load_metadata(path+"/metadata.json")
        except:
            raise Exception("error in loading the dataset:" + path)
