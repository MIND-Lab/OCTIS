import json
import gensim
import re
import spacy
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import multiprocessing as mp


stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def save(file, filename):
    with open(filename, 'w') as outfile:
        json.dump(file, outfile)


def load(filename):
    with open(filename) as json_file:
        result = json.load(json_file)
    return result

def createPool(nCpu):
     pool = mp.Pool(processes=nCpu)
     return pool

def multiprocess(pool, nCpu, function, corpus, arguments = []):
    tot = len(corpus)
    num = round(tot/nCpu) + 1
    tmpTot = tot
    totinv = 0
    ranges = [0]
    for _ in range(nCpu):
        if tmpTot - num >= 0:
            tmpTot -= num
            totinv += num
            ranges.append(totinv)
        else:
            ranges.append(tot)
    results = [pool.apply(function, args=(corpus[ranges[y]:ranges[y+1]], arguments)) for y in range(nCpu)]
    for y in range(nCpu):
        corpus[y*num:y*num +num] = results[y]
    return corpus


def removePunctuation(corpus, _):
    return removePunct(corpus)

def removePunct(corpus):
    corpus = [re.sub(r'\S*@\S*\s?', '', doc) for doc in corpus]
    corpus = [re.sub(r'\s+', ' ', doc) for doc in corpus]
    corpus = [re.sub(r"\'", "", doc) for doc in corpus]
    return corpus


def createBagsOfWords(corpus):
    for doc in corpus:
        yield(gensim.utils.simple_preprocess(str(doc), deacc=True))


def remove_stopwords(corpus,_):
    return remove_sw(corpus)

def remove_sw(corpus):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in corpus]


def lemmatization(corpus, arguments):
    result = []
    for document in corpus:
            doc = arguments[0](" ".join(document)) 
            result.append([token.lemma_ for token in doc if token.pos_ in arguments[1]])
    return result


def RemoveWords(corpus, minPar, maxPar):

    wordsDict = {}
    for document in corpus:
        FoundInArticle = {}
        for word in document:
            if word in wordsDict:
                if  not word in FoundInArticle:
                    wordsDict[word] += 1
                    FoundInArticle[word] = True
            else:
                wordsDict[word] = 1
                FoundInArticle[word] = True
    toRemove = {}
    for key, value in wordsDict.items():
        if value < minPar or value > maxPar:
            toRemove[key] = True
    return toRemove


def remove(corpus, keys):
    result = []
    for document in corpus:
        result.append([par for par in document if not par in keys])
    return result


def removeDocs(corpus, labels, minDoc):
    n = 0
    newCorpus = []
    newLabels = []
    wordsMean = 0
    cate = {}
    for document in corpus:
        lenDoc = len(document)
        if lenDoc > minDoc:
            wordsMean += lenDoc
            newCorpus.append(document)
            newLabels.append(labels[n])
            for label in labels[n]:
                if not label in cate:
                    cate[label] = True
            n += 1
    result = {}
    result["corpus"] = newCorpus
    result["docLabels"] = newLabels
    result["totalDocuments"] = n
    result["meanWordsPerDocument"] = round(wordsMean/n)
    result["labels"] = list(cate.keys())
    result["totalLabels"] = len(result["labels"])
    return result


def preprocessMultiprocess(dataset, minWordsForDoc, wordsMinFreq, wordsMaxFreq):
    content = dataset["corpus"]
    categories = dataset["docLabels"]
    print ("Creating pool")
    numproc = mp.cpu_count()
    pool = createPool(numproc)
    print("Pool created\nPreprocess initialization\n\n  removing punctuation")
    corpus = multiprocess(pool,numproc,removePunctuation,content)
    print("  Punctuation removed\n")
    corpusBag = list(createBagsOfWords(corpus))
    print("  Removing Stopwords")
    lp_nostops = multiprocess(pool,numproc,remove_stopwords,corpusBag)
    print("  Stopwords removed\n\n  lemmatizing")
    nlp = spacy.load('en', disable=['parser', 'ner'])
    data_lemmatized = multiprocess(pool, numproc, lemmatization, lp_nostops, [nlp, ['NOUN', 'ADJ', 'VERB', 'ADV']])
    print("  Lemmatized\n\n  Removing words that are present in less than "+str(wordsMinFreq)+" documents")
    print(" and im more than "+str(wordsMinFreq)+" documents")
    toRemove = RemoveWords(data_lemmatized, wordsMinFreq, wordsMaxFreq)
    removedWords = multiprocess(pool,numproc,remove,data_lemmatized,toRemove)
    print("  Words removed\n\n  Removing documents with less then "+str(minWordsForDoc)+" words")
    result = removeDocs(removedWords, categories, minWordsForDoc)
    print("  Documents removed\n\nPreprocess done!")
    pool.close()
    return result


def preprocess(dataset, minWordsForDoc, wordsMinFreq, wordsMaxFreq):
    content = dataset["corpus"]
    categories = dataset["docLabels"]
    print("Preprocess initialization\n\n  removing punctuation")
    corpus = removePunct(content)
    print("  Punctuation removed\n")
    corpusBag = list(createBagsOfWords(corpus))
    print("  Removing Stopwords")
    lp_nostops =remove_sw(corpusBag)
    print("  Stopwords removed\n\n  lemmatizing")
    nlp = spacy.load('en', disable=['parser', 'ner'])
    data_lemmatized = lemmatization(lp_nostops, [nlp, ['NOUN', 'ADJ', 'VERB', 'ADV']])
    print("  Lemmatized\n\n  Removing words that are present in less than "+str(wordsMinFreq)+" documents")
    print(" and im more than "+str(wordsMinFreq)+" documents")
    toRemove = RemoveWords(data_lemmatized, wordsMinFreq, wordsMaxFreq)
    removedWords = remove(data_lemmatized,toRemove)
    print("  Words removed\n\n  Removing documents with less then "+str(minWordsForDoc)+" words")
    result = removeDocs(removedWords, categories, minWordsForDoc)
    print("  Documents removed\n\nPreprocess done!")
    return result