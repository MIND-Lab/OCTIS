#!/usr/bin/env python

"""Tests for `octis` package."""

import pytest

from click.testing import CliRunner

from octis.dataset.dataset import Dataset
from octis.models.LDA import LDA
from octis.models.LDA_tomopy import LDA_tomopy as LDATOMOTO
from octis.models.ETM import ETM
from octis.models.CTM import CTM
from octis.models.NMF import NMF
from octis.models.NMF_scikit import NMF_scikit
from octis.models.ProdLDA import ProdLDA
from octis.preprocessing.preprocessing import Preprocessing

import numpy as np
import os


@pytest.fixture
def root_dir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def data_dir(root_dir):
    return root_dir + "/../preprocessed_datasets/"


@pytest.fixture
def embeddings_dir(root_dir):
    return root_dir + "/../trained_embeddings/"


def test_model_output_lda(data_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir + '/M10')
    num_topics = 3
    model = LDA(num_topics=num_topics, iterations=5)
    output = model.train_model(dataset)
    assert 'topics' in output.keys()
    assert 'topic-word-matrix' in output.keys()
    assert 'test-topic-document-matrix' in output.keys()

    # check topics format
    assert type(output['topics']) == list
    assert len(output['topics']) == num_topics

    # check topic-word-matrix format
    assert type(output['topic-word-matrix']) == np.ndarray
    assert output['topic-word-matrix'].shape == (num_topics, len(dataset.get_vocabulary()))

    # check topic-document-matrix format
    assert type(output['topic-document-matrix']) == np.ndarray
    assert output['topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[0]))

    # check test-topic-document-matrix format
    assert type(output['test-topic-document-matrix']) == np.ndarray
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[2]))


def test_model_output_etm(data_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir + '/M10')
    num_topics = 3
    model = ETM(num_topics=num_topics, num_epochs=5)
    output = model.train_model(dataset)
    assert 'topics' in output.keys()
    assert 'topic-word-matrix' in output.keys()
    assert 'test-topic-document-matrix' in output.keys()

    # check topics format
    assert type(output['topics']) == list
    assert len(output['topics']) == num_topics

    # check topic-word-matrix format
    assert type(output['topic-word-matrix']) == np.ndarray
    assert output['topic-word-matrix'].shape == (num_topics, len(dataset.get_vocabulary()))

    # check topic-document-matrix format
    assert type(output['topic-document-matrix']) == np.ndarray
    assert output['topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[0]))

    # check test-topic-document-matrix format
    assert type(output['test-topic-document-matrix']) == np.ndarray
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[2]))

def test_model_output_etm_with_corpus_containing_single_word_document(data_dir):
    texts_path = data_dir+"/sample_texts/unprepr_docs.txt"
    p = Preprocessing(vocabulary=None, max_features=None, remove_punctuation=True,
                       lemmatize=False, stopword_list='english')
    dataset = p.preprocess_dataset(
        documents_path=texts_path,
    )
    num_topics = 3
    model = ETM(num_topics=num_topics, num_epochs=5)
    output = model.train_model(dataset)
    assert 'topics' in output.keys()
    assert 'topic-word-matrix' in output.keys()
    assert 'test-topic-document-matrix' in output.keys()

    # check topics format
    assert type(output['topics']) == list
    assert len(output['topics']) == num_topics

    # check topic-word-matrix format
    assert type(output['topic-word-matrix']) == np.ndarray
    assert output['topic-word-matrix'].shape == (num_topics, len(dataset.get_vocabulary()))

    # check topic-document-matrix format
    assert type(output['topic-document-matrix']) == np.ndarray
    assert output['topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[0]))

    # check test-topic-document-matrix format
    assert type(output['test-topic-document-matrix']) == np.ndarray
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[2]))


def test_model_output_etm_not_partitioned(data_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir + '/M10')
    num_topics = 3
    model = ETM(num_topics=num_topics, num_epochs=5, use_partitions=False)
    output = model.train_model(dataset)
    assert 'topics' in output.keys()
    assert 'topic-word-matrix' in output.keys()
    assert 'test-topic-document-matrix' not in output.keys()

    # check topics format
    assert type(output['topics']) == list
    assert len(output['topics']) == num_topics

    # check topic-word-matrix format
    assert type(output['topic-word-matrix']) == np.ndarray
    assert output['topic-word-matrix'].shape == (num_topics, len(dataset.get_vocabulary()))

    # check topic-document-matrix format
    assert type(output['topic-document-matrix']) == np.ndarray
    assert output['topic-document-matrix'].shape == (num_topics, len(dataset.get_corpus()))


def test_model_output_etm_with_pickle_embeddings_file(data_dir, embeddings_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir + '/M10')
    num_topics = 3
    model = ETM(num_topics=num_topics, num_epochs=5, train_embeddings=False, 
        embeddings_path=embeddings_dir +'/test_example/example.pickle')
    output = model.train_model(dataset)
    assert 'topics' in output.keys()
    assert 'topic-word-matrix' in output.keys()
    assert 'test-topic-document-matrix' in output.keys()

    # check topics format
    assert type(output['topics']) == list
    assert len(output['topics']) == num_topics

    # check topic-word-matrix format
    assert type(output['topic-word-matrix']) == np.ndarray
    assert output['topic-word-matrix'].shape == (num_topics, len(dataset.get_vocabulary()))

    # check topic-document-matrix format
    assert type(output['topic-document-matrix']) == np.ndarray
    assert output['topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[0]))

    # check test-topic-document-matrix format
    assert type(output['test-topic-document-matrix']) == np.ndarray
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[2]))

def test_model_output_etm_with_binary_word2vec_embeddings_file(data_dir, embeddings_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir + '/M10')
    num_topics = 3
    model = ETM(num_topics=num_topics, num_epochs=5, train_embeddings=False, 
        embeddings_type='word2vec', embeddings_path=embeddings_dir +'/test_example/example.bin')
    output = model.train_model(dataset)
    assert 'topics' in output.keys()
    assert 'topic-word-matrix' in output.keys()
    assert 'test-topic-document-matrix' in output.keys()

    # check topics format
    assert type(output['topics']) == list
    assert len(output['topics']) == num_topics

    # check topic-word-matrix format
    assert type(output['topic-word-matrix']) == np.ndarray
    assert output['topic-word-matrix'].shape == (num_topics, len(dataset.get_vocabulary()))

    # check topic-document-matrix format
    assert type(output['topic-document-matrix']) == np.ndarray
    assert output['topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[0]))

    # check test-topic-document-matrix format
    assert type(output['test-topic-document-matrix']) == np.ndarray
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[2]))

def test_model_output_etm_with_text_word2vec_embeddings_file(data_dir, embeddings_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir + '/M10')
    num_topics = 3
    model = ETM(num_topics=num_topics, num_epochs=5, train_embeddings=False, 
        embeddings_type='word2vec', embeddings_path=embeddings_dir +'/test_example/example.txt', 
        binary_embeddings=False)
    output = model.train_model(dataset)
    assert 'topics' in output.keys()
    assert 'topic-word-matrix' in output.keys()
    assert 'test-topic-document-matrix' in output.keys()

    # check topics format
    assert type(output['topics']) == list
    assert len(output['topics']) == num_topics

    # check topic-word-matrix format
    assert type(output['topic-word-matrix']) == np.ndarray
    assert output['topic-word-matrix'].shape == (num_topics, len(dataset.get_vocabulary()))

    # check topic-document-matrix format
    assert type(output['topic-document-matrix']) == np.ndarray
    assert output['topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[0]))

    # check test-topic-document-matrix format
    assert type(output['test-topic-document-matrix']) == np.ndarray
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[2]))

def test_model_output_etm_with_headerless_text_word2vec_embeddings_file(data_dir, embeddings_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir + '/M10')
    num_topics = 3
    model = ETM(num_topics=num_topics, num_epochs=5, train_embeddings=False, 
        embeddings_type='word2vec', embeddings_path=embeddings_dir +'/test_example/headerless_example.txt', 
        binary_embeddings=False, headerless_embeddings=True)
    output = model.train_model(dataset)
    assert 'topics' in output.keys()
    assert 'topic-word-matrix' in output.keys()
    assert 'test-topic-document-matrix' in output.keys()

    # check topics format
    assert type(output['topics']) == list
    assert len(output['topics']) == num_topics

    # check topic-word-matrix format
    assert type(output['topic-word-matrix']) == np.ndarray
    assert output['topic-word-matrix'].shape == (num_topics, len(dataset.get_vocabulary()))

    # check topic-document-matrix format
    assert type(output['topic-document-matrix']) == np.ndarray
    assert output['topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[0]))

    # check test-topic-document-matrix format
    assert type(output['test-topic-document-matrix']) == np.ndarray
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[2]))

def test_model_output_etm_with_keyedvectors_embeddings_file(data_dir, embeddings_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir + '/M10')
    num_topics = 3
    model = ETM(num_topics=num_topics, num_epochs=5, train_embeddings=False, 
        embeddings_type='keyedvectors', embeddings_path=embeddings_dir +'/test_example/example.keyedvectors')
    output = model.train_model(dataset)
    assert 'topics' in output.keys()
    assert 'topic-word-matrix' in output.keys()
    assert 'test-topic-document-matrix' in output.keys()

    # check topics format
    assert type(output['topics']) == list
    assert len(output['topics']) == num_topics

    # check topic-word-matrix format
    assert type(output['topic-word-matrix']) == np.ndarray
    assert output['topic-word-matrix'].shape == (num_topics, len(dataset.get_vocabulary()))

    # check topic-document-matrix format
    assert type(output['topic-document-matrix']) == np.ndarray
    assert output['topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[0]))

    # check test-topic-document-matrix format
    assert type(output['test-topic-document-matrix']) == np.ndarray
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[2]))

def test_model_output_lda_tomotopy(data_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir + '/M10')
    num_topics = 3
    model = LDATOMOTO(num_topics=num_topics, alpha=0.1)
    output = model.train_model(dataset)
    assert 'topics' in output.keys()
    assert 'topic-word-matrix' in output.keys()
    assert 'test-topic-document-matrix' in output.keys()

    # check topics format
    assert type(output['topics']) == list
    assert len(output['topics']) == num_topics

    # check topic-word-matrix format
    assert type(output['topic-word-matrix']) == np.ndarray
    assert output['topic-word-matrix'].shape == (num_topics, len(dataset.get_vocabulary()))

    # check topic-document-matrix format
    assert type(output['topic-document-matrix']) == np.ndarray
    assert output['topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[0]))

    # check test-topic-document-matrix format
    assert type(output['test-topic-document-matrix']) == np.ndarray
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[2]))


def test_model_output_nmf(data_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir + '/M10')
    num_topics = 3
    model = NMF(num_topics=num_topics, w_max_iter=10, h_max_iter=10, use_partitions=True)
    output = model.train_model(dataset)
    assert 'topics' in output.keys()
    assert 'topic-word-matrix' in output.keys()
    assert 'test-topic-document-matrix' in output.keys()

    # check topics format
    assert type(output['topics']) == list
    assert len(output['topics']) == num_topics

    # check topic-word-matrix format
    assert type(output['topic-word-matrix']) == np.ndarray
    assert output['topic-word-matrix'].shape == (num_topics, len(dataset.get_vocabulary()))

    # check topic-document-matrix format
    assert type(output['topic-document-matrix']) == np.ndarray
    assert output['topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[0]))

    # check test-topic-document-matrix format
    assert type(output['test-topic-document-matrix']) == np.ndarray
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[2]))


def test_model_output_nmf_scikit(data_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir + '/M10')
    num_topics = 3
    model = NMF_scikit(num_topics=num_topics, use_partitions=True)
    output = model.train_model(dataset)
    assert 'topics' in output.keys()
    assert 'topic-word-matrix' in output.keys()
    assert 'test-topic-document-matrix' in output.keys()

    # check topics format
    assert type(output['topics']) == list
    assert len(output['topics']) == num_topics

    # check topic-word-matrix format
    assert type(output['topic-word-matrix']) == np.ndarray
    assert output['topic-word-matrix'].shape == (num_topics, len(dataset.get_vocabulary()))

    # check topic-document-matrix format
    assert type(output['topic-document-matrix']) == np.ndarray
    assert output['topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[0]))

    # check test-topic-document-matrix format
    assert type(output['test-topic-document-matrix']) == np.ndarray
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[2]))


def test_model_output_ctm_zeroshot(data_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir + '/M10')
    num_topics = 3
    model = CTM(num_topics=num_topics, num_epochs=5, inference_type='zeroshot')
    output = model.train_model(dataset)
    assert 'topics' in output.keys()
    assert 'topic-word-matrix' in output.keys()
    assert 'test-topic-document-matrix' in output.keys()

    # check topics format
    assert type(output['topics']) == list
    assert len(output['topics']) == num_topics

    # check topic-word-matrix format
    assert type(output['topic-word-matrix']) == np.ndarray
    assert output['topic-word-matrix'].shape == (num_topics, len(dataset.get_vocabulary()))

    # check topic-document-matrix format
    assert type(output['topic-document-matrix']) == np.ndarray
    assert output['topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[0]))

    # check test-topic-document-matrix format
    assert type(output['test-topic-document-matrix']) == np.ndarray
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[2]))


def test_model_output_ctm_combined(data_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir + '/M10')
    num_topics = 3
    model = CTM(num_topics=num_topics, num_epochs=5, inference_type='combined')
    output = model.train_model(dataset)
    assert 'topics' in output.keys()
    assert 'topic-word-matrix' in output.keys()
    assert 'test-topic-document-matrix' in output.keys()

    # check topics format
    assert type(output['topics']) == list
    assert len(output['topics']) == num_topics

    # check topic-word-matrix format
    assert type(output['topic-word-matrix']) == np.ndarray
    assert output['topic-word-matrix'].shape == (num_topics, len(dataset.get_vocabulary()))

    # check topic-document-matrix format
    assert type(output['topic-document-matrix']) == np.ndarray
    assert output['topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[0]))

    # check test-topic-document-matrix format
    assert type(output['test-topic-document-matrix']) == np.ndarray
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[2]))


def test_model_output_ctm_combined_not_partition(data_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir + '/M10')
    num_topics = 3
    model = CTM(num_topics=num_topics, num_epochs=5, inference_type='combined',use_partitions=False,
                bert_path='./not_part')
    output = model.train_model(dataset)
    assert 'topics' in output.keys()
    assert 'topic-word-matrix' in output.keys()
    assert 'test-topic-document-matrix' not in output.keys()

    # check topics format
    assert type(output['topics']) == list
    assert len(output['topics']) == num_topics

    # check topic-word-matrix format
    assert type(output['topic-word-matrix']) == np.ndarray
    assert output['topic-word-matrix'].shape == (num_topics, len(dataset.get_vocabulary()))

    # check topic-document-matrix format
    assert type(output['topic-document-matrix']) == np.ndarray
    assert output['topic-document-matrix'].shape == (num_topics, len(dataset.get_corpus()))


def test_model_output_prodlda(data_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir + '/M10')
    num_topics = 3
    model = ProdLDA(num_topics=num_topics, num_epochs=5)
    output = model.train_model(dataset)
    assert 'topics' in output.keys()
    assert 'topic-word-matrix' in output.keys()
    assert 'test-topic-document-matrix' in output.keys()

    # check topics format
    assert type(output['topics']) == list
    assert len(output['topics']) == num_topics

    # check topic-word-matrix format
    assert type(output['topic-word-matrix']) == np.ndarray
    assert output['topic-word-matrix'].shape == (num_topics, len(dataset.get_vocabulary()))

    # check topic-document-matrix format
    assert type(output['topic-document-matrix']) == np.ndarray
    assert output['topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[0]))

    # check test-topic-document-matrix format
    assert type(output['test-topic-document-matrix']) == np.ndarray
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[2]))


def test_model_output_prodlda_not_partitioned(data_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir + '/M10')
    num_topics = 3
    model = ProdLDA(num_topics=num_topics, num_epochs=5, use_partitions=False)
    output = model.train_model(dataset)
    assert 'topics' in output.keys()
    assert 'topic-word-matrix' in output.keys()
    assert 'test-topic-document-matrix' not in output.keys()

    # check topics format
    assert type(output['topics']) == list
    assert len(output['topics']) == num_topics

    # check topic-word-matrix format
    assert type(output['topic-word-matrix']) == np.ndarray
    assert output['topic-word-matrix'].shape == (num_topics, len(dataset.get_vocabulary()))

    # check topic-document-matrix format
    assert type(output['topic-document-matrix']) == np.ndarray
    assert output['topic-document-matrix'].shape == (num_topics, len(dataset.get_corpus()))

