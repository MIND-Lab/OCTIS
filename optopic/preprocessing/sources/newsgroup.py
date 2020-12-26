import gensim.downloader as gd
from sklearn.datasets import fetch_20newsgroups


def retrieve_20newsgroup_gensim():
    """
    Retrieve the corpus and the labels

    Returns
    -------
    result : dictionary with corpus, training and test partition
             and labels of the corpus
    """
    dataset = gd.load("20-newsgroups")
    corpus = []
    labels = []
    partition = None

    for data in dataset:
        corpus.append(data["data"])
        if data["set"] == "test" and partition is None:
            partition = len(corpus) - 1
        labels.append([data["topic"]])
    result = dict()
    result["partition"] = partition
    result["corpus"] = corpus
    result["doc_labels"] = labels
    result["info"] = {
        "name": "20-newsgroups",
        "link": "http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz",
        "source": "https://radimrehurek.com/gensim/",
        "num_records": 18846,
        "description": "The notorious collection of approximately 20,000 newsgroup posts, partitioned (nearly) evenly across 20 different newsgroups.",
        "file_name": "20-newsgroups.gz",
        "info": "http://qwone.com/~jason/20Newsgroups/"
    }
    return result


def retrieve_20newsgroup_scikit():
    """
    Retrieve the corpus and the labels

    Returns
    -------
    result : dictionary with corpus, training and test partition
             and labels of the corpus
    """
    corpus = []
    labels = []

    newsgroup = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

    cate = newsgroup.target_names

    corpus = corpus + newsgroup.data

    for doc in newsgroup.target:
        labels.append([cate[doc]])

    partition = len(corpus) - 1

    result = dict()
    result["partition"] = partition
    result["corpus"] = corpus
    result["doc_labels"] = labels
    result["info"] = {
        "name": "20-newsgroups",
        "link": "http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz",
        "source": "https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html",
        "num_records": 18846,
        "description": "The notorious collection of approximately 20,000 newsgroup posts, partitioned (nearly) evenly across 20 different newsgroups.",
        "file_name": "20-newsgroups.gz",
        "info": "http://qwone.com/~jason/20Newsgroups/"
    }

    return result
