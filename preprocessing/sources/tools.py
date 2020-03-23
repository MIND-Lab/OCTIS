from collections import namedtuple
import gensim
import gensim.utils as ut
from pathlib import Path


network_sentence=namedtuple('NetworkSentence', 'words tags labels index')

def __read_network_data(dir, stemmer=0):
    all_index={}
    all_docs = []
    label_set = set()
    with open(dir+'/docs.txt') as f1, open(dir + '/labels.txt') as f2:

        for l1 in f1:
            if stemmer == 1:
                l1 = gensim.parsing.stem_text(l1)
            else:
                l1 = l1.lower()
            tokens = ut.to_unicode(l1).split()

            words = tokens[1:]
            tags = [tokens[0]] 
            index = len(all_docs)
            all_index[tokens[0]] = index 

            l2 = f2.readline()
            tokens2 = gensim.utils.to_unicode(l2).split()
            labels = tokens2[1]
            label_set.add(labels)
            all_docs.append(network_sentence(words, tags, labels, index))

    return all_docs, all_index, list(label_set)


# Retrieve the dataset and the labels
def _retrieve(ds):
    directory = str(Path(__file__).parent.absolute())+'/datasets/'+ds
    alldocs, _ , _ = __read_network_data(directory)
    corpus = []
    labels = []
    for doc in alldocs:
        doc_with_spaces = " ".join(doc.words)
        corpus.append(doc_with_spaces)
        labels.append([doc.labels])
    result = {}
    result["corpus"] = corpus
    result["doc_labels"] = labels
    return result
