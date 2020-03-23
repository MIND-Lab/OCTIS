from collections import namedtuple
import gensim
import gensim.utils as ut
from pathlib import Path


NetworkSentence=namedtuple('NetworkSentence', 'words tags labels index')

def readNetworkData(dir, stemmer=0): #dir, directory of network dataset
    allindex={}
    alldocs = []
    labelset = set()
    with open(dir+'/docs.txt') as f1, open(dir + '/labels.txt') as f2:

        for l1 in f1:
            #tokens = ut.to_unicode(l1.lower()).split()
            if stemmer == 1:
                l1 = gensim.parsing.stem_text(l1)
            else:
                l1 = l1.lower()
            tokens = ut.to_unicode(l1).split()

            words = tokens[1:]
            tags = [tokens[0]] # ID of each document, for doc2vec model
            index = len(alldocs)
            allindex[tokens[0]] = index # A mapping from documentID to index, start from 0

            l2 = f2.readline()
            tokens2 = gensim.utils.to_unicode(l2).split()
            labels = tokens2[1] #class label
            labelset.add(labels)
            alldocs.append(NetworkSentence(words, tags, labels, index))

    return alldocs, allindex, list(labelset)

def retrieve(ds):
    directory = str(Path(__file__).parent.absolute())+'/Datasets/'+ds
    alldocs, _ , _ = readNetworkData(directory)
    corpus = []
    labels = []
    for doc in alldocs:
        docWithSpaces = " ".join(doc.words)
        corpus.append(docWithSpaces)
        labels.append([doc.labels])
    result = {}
    result["corpus"] = corpus
    result["docLabels"] = labels
    return result
