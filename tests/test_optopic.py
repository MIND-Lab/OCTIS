#!/usr/bin/env python

"""Tests for `octis` package."""

import pytest

from click.testing import CliRunner

from octis.dataset.dataset import Dataset
from octis.models.LDA import LDA
from octis.models.ETM import ETM
from octis.models.CTM import CTM
from octis.models.NMF import NMF
from octis.models.NMF_scikit import NMF_scikit
from octis.models.ProdLDA import ProdLDA

import numpy as np
import os


@pytest.fixture
def root_dir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def data_dir(root_dir):
    return root_dir + "/../preprocessed_datasets/"


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
