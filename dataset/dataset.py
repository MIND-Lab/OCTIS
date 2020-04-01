import json 
from pathlib import Path


class Dataset:
    
    def __init__(self, corpus = [], vocabulary = {}, metadata = {}, labels = []):
        self.set_corpus(corpus)
        self.set_vocabulary(vocabulary)
        self.set_metadata(metadata)
        self.set_labels(labels)


    def set_corpus(self, corpus):
        self.__corpus = corpus

    def get_corpus(self):
        if self.__corpus != []:
            return self.__corpus
        return False


    def set_labels(self, labels):
        self.__labels = labels

    def get_labels(self):
        if self.__labels != []:
            return self.__labels
        return False


    def set_metadata(self, metadata):
        self.__metadata = metadata

    def get_metadata(self):
        if self.__metadata != {}:
            return self.__metadata
        return False


    def set_vocabulary(self, vocabulary):
        self.__vocabulary = vocabulary

    def get_vocabulary(self):
        if self.__vocabulary != {}:
            return self.__vocabulary
        return False


    # Saves metadata in json serialized format
    def _save_metadata(self, file_name):
        data = self.get_metadata()
        if data:
            with open(file_name, 'w') as outfile:
                json.dump(data, outfile)
            return True

    def _load_metadata(self, file_name):
        metadata = {}
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as metadata_file:
                metadata = json.load(metadata_file)
            self.set_metadata(metadata)
            return True
        return False


    # Saves the corpus, a document for each line
    def _save_corpus(self, file_name):
        data = self.get_corpus()
        if data:
            with open(file_name, 'w') as outfile:
                for element in data:
                    outfile.write("%s\n" % " ".join(element))
            return True

    def _load_corpus(self, file_name):
        corpus = []
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as corpus_file:
                for line in corpus_file:
                    corpus.append(line.split())
            self.set_corpus(corpus)
            return True
        return False


    # Saves the labels, each line contains the labels of a single document
    def _save_labels(self, file_name):
        data = self.get_labels()
        if data:
            with open(file_name, 'w') as outfile:
                for element in data:
                    outfile.write("%s\n" % json.dumps(element))
            return True

    def _load_labels(self, file_name):
        labels = []
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as labels_file:
                for line in labels_file:
                    labels.append(json.loads(line))
            self.set_labels(labels)
            return True
        return False


    # Saves the vocabulary in list format
    def _save_vocabulary(self, file_name):
        data = self.get_vocabulary()
        if data:
            with open(file_name, 'w') as outfile:
                for  word, freq in data.items():
                    line = word+" "+str(freq)
                    outfile.write("%s\n" % line)
            return True

    def _load_vocabulary(self, file_name):
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

    # Saves all corpus, vocabulary and metadata
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._save_corpus(path+"/corpus.txt")
        self._save_vocabulary(path+"/vocabulary.txt")
        self._save_labels(path+"/labels.txt")
        self._save_metadata(path+"/metadata.json")
        return True

    def load(self, path):
        self._load_corpus(path+"/corpus.txt")
        self._load_vocabulary(path+"/vocabulary.txt")
        self._load_labels(path+"/labels.txt")
        self._load_metadata(path+"/metadata.json")
        return True
