import codecs
import json
import pickle
from os.path import join, exists
from pathlib import Path

import pandas as pd
from werkzeug.utils import header_property

from octis.dataset.downloader import get_data_home, _pkl_filepath, download_dataset


class Dataset:
    """
    Dataset handles a dataset and offers methods to access, save and edit the dataset data
    """

    def __init__(self, corpus=None, vocabulary=None, labels=None, metadata=None, document_indexes=None):
        """
        Initialize a dataset, parameters are optional
        if you want to load a dataset, initialize this
        class with default values and use the load method
        Parameters
        ----------
        corpus : corpus of the dataset
        vocabulary : vocabulary of the dataset
        labels : labels of the dataset
        metadata : metadata of the dataset
        """
        self.__corpus = corpus
        self.__vocabulary = vocabulary
        self.__metadata = metadata
        self.__labels = labels
        self.__original_indexes = document_indexes
        self.dataset_path = None
        self.is_cached = False

    def get_corpus(self):
        return self.__corpus

    # Partitioned Corpus getter
    def get_partitioned_corpus(self, use_validation=True):
        last_training_doc = self.__metadata["last-training-doc"]
        # gestire l'eccezione se last_validation_doc non è definito, restituire
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
                edges = [line[0:len(line) - 1] for line in edges_file]
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
                for word in data:
                    outfile.write(word + "\n")
        else:
            raise Exception("error in saving vocabulary")

    def _save_document_indexes(self, file_name):
        """
        Saves document indexes in a file
        Parameters
        ----------
        file_name : name of the file to write
        -------
        """
        if self.__original_indexes is not None:
            with open(file_name, 'w') as outfile:
                for i in self.__original_indexes:
                    outfile.write(str(i) + "\n")

    def _load_vocabulary(self, file_name):
        """
        Loads vocabulary from a file
        Parameters
        ----------
        file_name : name of the file to read
        """
        vocabulary = []
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as vocabulary_file:
                for line in vocabulary_file:
                    vocabulary.append(line.strip())
            self.__vocabulary = vocabulary
        else:
            raise Exception("error in loading vocabulary")

    def _load_document_indexes(self, file_name):
        """
        Loads document indexes from a file
        Parameters
        ----------
        file_name : name of the file to read
        """
        document_indexes = []
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as indexes_file:
                for line in indexes_file:
                    document_indexes.append(line.strip())
            self.__original_indexes = document_indexes
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
            partitions = self.get_partitioned_corpus()
            corpus, partition = [], []
            for i, p in enumerate(partitions):
                if i == 0:
                    part = 'train'
                elif i == 1 and len(partitions) == 3:
                    part = 'val'
                else:
                    part = 'test'

                for doc in p:
                    corpus.append(' '.join(doc))
                    partition.append(part)

            df = pd.DataFrame(data=corpus)
            df = pd.concat([df, pd.DataFrame(partition)], axis=1)
            if self.__labels:
                df = pd.concat([df, pd.DataFrame(self.__labels)], axis=1)
            df.to_csv(path + '/corpus.tsv', sep='\t', index=False, header=None)

            self._save_vocabulary(path + "/vocabulary.txt")
            self._save_metadata(path + "/metadata.json")
            self._save_document_indexes(path + "/indexes.txt")
            self.dataset_path = path

        except:
            raise Exception("error in saving the dataset")

    def load_custom_dataset_from_folder(self, path):
        """
        Loads all the dataset from a folder
        Parameters
        ----------
        path : path of the folder to read
        """
        self.dataset_path = path
        try:
            if exists(self.dataset_path + "/metadata.json"):
                self._load_metadata(self.dataset_path + "/metadata.json")
            else:
                self.__metadata = dict()
            df = pd.read_csv(self.dataset_path + "/corpus.tsv", sep='\t', header=None)
            if len(df.keys()) > 1:
                df[1] = df[1].replace("train", "a_train")
                df[1] = df[1].replace("val", "b_val")
                df = df.sort_values(1).reset_index(drop=True)

                self.__metadata['last-training-doc'] = len(df[df[1] == 'a_train'])
                self.__metadata['last-validation-doc'] = len(df[df[1] == 'b_val']) + len(df[df[1] == 'a_train'])

                self.__corpus = [d.split() for d in df[0].tolist()]
                if len(df.keys()) > 2:
                    self.__labels = df[2].tolist()
            else:
                self.__corpus = [d.split() for d in df[0].tolist()]
                self.__metadata['last-training-doc'] = len(df[0])

            if exists(self.dataset_path + "/vocabulary.txt"):
                self._load_vocabulary(self.dataset_path + "/vocabulary.txt")
            else:
                vocab = set()
                for d in self.__corpus:
                    for w in set(d):
                        vocab.add(w)
                self.__vocabulary = list(vocab)
            if exists(self.dataset_path + "/indexes.txt"):
                self._load_document_indexes(self.dataset_path + "/indexes.txt")
        except:
            raise Exception("error in loading the dataset:" + self.dataset_path)

    def fetch_dataset(self, dataset_name, data_home=None, download_if_missing=True):
        """Load the filenames and data from a dataset.
        Parameters
        ----------
        dataset_name: name of the dataset to download or retrieve
        data_home : optional, default: None
            Specify a download and cache folder for the datasets. If None,
            all data is stored in '~/octis' subfolders.
        download_if_missing : optional, True by default
            If False, raise an IOError if the data is not locally available
            instead of trying to download the data from the source site.
        """

        data_home = get_data_home(data_home=data_home)
        cache_path = _pkl_filepath(data_home, dataset_name + ".pkz")
        dataset_home = join(data_home, dataset_name)
        cache = None
        if exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    compressed_content = f.read()
                uncompressed_content = codecs.decode(
                    compressed_content, 'zlib_codec')
                cache = pickle.loads(uncompressed_content)
            except Exception as e:
                print(80 * '_')
                print('Cache loading failed')
                print(80 * '_')
                print(e)

        if cache is None:
            if download_if_missing:
                cache = download_dataset(dataset_name, target_dir=dataset_home, cache_path=cache_path)
            else:
                raise IOError(dataset_name + ' dataset not found')
        self.is_cached = True
        self.__corpus = [d.split() for d in cache["corpus"]]
        self.__vocabulary = cache["vocabulary"]
        self.__metadata = cache["metadata"]
        self.__labels = cache["labels"]

