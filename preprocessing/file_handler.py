import json 
from pathlib import Path


# Saves metadata in json serialized format
def _save_metadata(metadata, file_name):
    with open(file_name, 'w') as outfile:
        json.dump(metadata, outfile)
    return True


# Saves the corpus, a document for each line
def _save_corpus(data, file_name):
    with open(file_name, 'w') as outfile:
        for element in data:
            outfile.write("%s\n" % " ".join(element))
    return True


# Saves the labels, each line contains the labels of a single document
def _save_labels(data, file_name):
    with open(file_name, 'w') as outfile:
        for element in data:
            outfile.write("%s\n" % json.dumps(element))
    return True


# Saves the vocabulary in list format
def _save_vocabulary(vocabulary, file_name):
    with open(file_name, 'w') as outfile:
        for  word, freq in vocabulary.items():
            line = word+" "+str(freq)
            outfile.write("%s\n" % line)


# Saves all corpus, vocabulary and metadata
def save(data, path):
    Path(path).mkdir(parents=True, exist_ok=True)
    if "corpus" in data:
        _save_corpus(data["corpus"], path+"/corpus.txt")
    if "vocabulary" in data:
        _save_vocabulary(data["vocabulary"], path+"/vocabulary.txt")
    if "labels" in data:
        _save_labels(data["labels"], path+"/labels.txt")
    if "metadata" in data:
        _save_metadata(data["metadata"], path+"/metadata.json")
    return True
