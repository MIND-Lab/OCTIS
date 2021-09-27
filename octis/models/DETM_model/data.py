import os
import random
import pickle
import pickle
import numpy as np
import torch 
import scipy.io
from pathlib import Path
from gensim.models.fasttext import FastText as FT_gensim
from tqdm import tqdm
from scipy.io import loadmat
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, hstack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_mat_file(key, path):
    """
    read the preprocess mat file whose key and and path are passed as parameters

    Args:
        key ([type]): [description]
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    term_path = Path().cwd().joinpath('data', 'preprocess', path)
    doc = loadmat(term_path)[key]
    return doc


def get_time_columns(matrix):
    """
    given a matrix where the time column is the last row return the doc term matrix and the times
    """
    doc_term_matrix = matrix[:, :-1]
    times = matrix[:, -1].toarray().astype(np.int32).squeeze()
    return doc_term_matrix, times


def split_train_test_matrix(dataset):
    """Split the dataset into the train set , the validation and the test set

    Args:
        dataset ([type]): [description]

    Returns:
        [type]: [description]
    """
    X_train, X_test_global = train_test_split(dataset, test_size=0.25, random_state=1)
    X_val, X_test = train_test_split(X_test_global, test_size=0.5, random_state=1)
    X_test_1, X_test_2 = train_test_split(X_test, test_size=0.5, random_state=1)
    X_test_val, X_test = train_test_split(X_val, test_size=0.5, random_state=1)
    return X_train, X_val, X_test_1, X_test_2, X_test


def get_data(doc_terms_file_name="tf_idf_doc_terms_matrix",
             terms_filename="tf_idf_terms",
             timestamp_filename='tf_idf_timestamps'):
    """read the data and return the vocabulary as well as the train, test and validation tests

    Args:
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    doc_term_matrix = read_mat_file("doc_terms_matrix", doc_terms_file_name)
    terms = read_mat_file("terms", terms_filename)
    times = read_mat_file("timestamps", timestamp_filename)
    vocab = terms
    #  make the time the last row of the doc_terms matrix for better spliting
    times = times.reshape((times.shape[1], 1))
    doc_term_time_matrix = hstack([doc_term_matrix, times])
    train, validation, test_1, test_2, test = split_train_test_matrix(doc_term_time_matrix)
    return vocab, train, validation, test_1, test_2, test


def get_batch(doc_terms_matrix, indices, device, timestamps):
    """
    get the a sample of the given indices

    Basically get the given indices from the dataset

    Args:
        doc_terms_matrix ([type]): the document term matrix
        indices ([type]):  numpy array
        vocab_size ([type]): [description]

    Returns:
        [numpy array]: a numpy array with the data passed as parameter
    """
    data_batch = doc_terms_matrix[indices, :]
    data_batch = torch.from_numpy(data_batch.toarray()).float().to(device)
    times_batch = timestamps[indices]
    return data_batch, times_batch


def train_or_load(args1, args2):
    """This decorator load the data passed in parameter or return the train function

    Args:
        func ([type]): [description]
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if args[3] in ["train", "test", "valid", "test_1", "test_2"]:
                data_path = args1.joinpath(f"rnn_input_{args[3]}.pkl")
                kwargs['data_path'] = data_path
            if args2:
                dataset = torch.load(data_path)
                return dataset
            return func(*args, **kwargs)
        return wrapper
    return decorator


@train_or_load(Path.cwd().joinpath("data", "rnn_inputs"), True)
def get_rnn_input(doc_terms_matrix, num_times, vocab_size, mode, **kwargs):
    """
    need to understand this part

    Args:
        doc_terms_matrix ([type]): [description]
        counts ([type]): [description]
        timestamps ([type]): [description]
        num_times ([type]): [description]
        vocab_size ([type]): [description]
        num_docs ([type]): [description]

    Returns:
        [type]: [description]
    """
    doc_terms_matrix, timestamps = get_time_columns(doc_terms_matrix)
    num_docs = doc_terms_matrix.shape[0]
    indices = torch.randperm(num_docs)
    indices = torch.split(indices, 1000)
    rnn_input = torch.zeros(num_times, vocab_size).to(device)
    cnt = torch.zeros(num_times, ).to(device)
    for idx, ind in tqdm(enumerate(indices)):
        data_batch, times_batch = get_batch(doc_terms_matrix, ind, device, timestamps)
        # this part I will look into it after
        for t in range(num_times):
            tmp = (times_batch == t).nonzero()
            docs = data_batch[tmp].squeeze().sum(0)
            rnn_input[t] += docs
            cnt[t] += len(tmp)
        if idx % 20 == 0:
            print('idx: {}/{}'.format(idx, len(indices)))
    rnn_input = rnn_input / cnt.unsqueeze(1)
    torch.save(rnn_input, kwargs.get("data_path"))
    return rnn_input


@train_or_load(Path().cwd().joinpath('data', 'preprocess', "embedding_matrix.npy"), True)
def read_embedding_matrix(vocab, device,  data_path):
    """
    read the embedding  matrix passed as parameter and return it as an vocabulary of each word
    with the corresponding embeddings

    Args:
        path ([type]): [description]

    # we need to use tensorflow embedding lookup heer
    """
    model_path = Path.cwd().joinpath("octis",
                                     "models",
                                     "ETM_model",
                                     "data",
                                     'embeddings_one_gram_fast_tweets_only').__str__()
    embeddings_path = Path().cwd().joinpath('data', 'preprocess', "embedding_matrix.npy")
    model_gensim = FT_gensim.load(model_path)
    vectorized_get_embeddings = np.vectorize(model_gensim.wv.get_vector)
    embeddings_matrix = np.zeros(shape=(len(vocab),50)) #should put the embeding size as a vector
    print("starting getting the word embeddings ++++ ")
    vocab = vocab.ravel()
    for index, word in tqdm(enumerate(vocab)):
        vector = model_gensim.wv.get_vector(word)
        embeddings_matrix[index] = vector
    print("done getting the word embeddings ")
    with open(embeddings_path, 'wb') as file_path:
        np.save(file_path, embeddings_matrix)
    embeddings = torch.from_numpy(embeddings_matrix).to(device)
    return embeddings
