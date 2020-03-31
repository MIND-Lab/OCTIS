Dataset nomenclature:
    name_lemmatized_min
or, if the dataset is not lemmatized:
    name_min

name = name of the original dataset
min = min %*10 of documents in wich each word must appear

Dataset standard:
A dataset is composed of a folder with up to 4 files:
corpus.txt = contains the corpus, each line is a document.
labels.txt = contains the topics each document covers.
             Each line refer to a topic.
             The labels of a document are serialized in a list.
metadata.json = contains extra info about the dataset
vocabulary.txt = contains the words used in the dataset and their
                 frequency in the documents of the corpus.