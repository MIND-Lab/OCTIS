#!/usr/bin/env python

"""Tests for `octis` package."""

import pytest

from click.testing import CliRunner

from octis.evaluation_metrics.coherence_metrics import *
from octis.dataset.dataset import Dataset
from octis.models.LDA import LDA
from octis.models.ETM import ETM
from octis.models.CTM import CTM

import os


@pytest.fixture
def root_dir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def data_dir(root_dir):
    return root_dir + "/../octis/preprocessed_datasets/"


def test_coherence_measures(data_dir):
    dataset = Dataset()
    dataset.load(data_dir + '/M10')

    model = LDA(num_topics=3, iterations=5)
    output = model.train_model(dataset)
    metrics_parameters = {'topk': 10, "texts": dataset.get_corpus()}
    metric = Coherence(metrics_parameters)
    score = metric.score(output)
    assert type(score) == np.float64


def test_model_output_lda(data_dir):
    dataset = Dataset()
    dataset.load(data_dir + '/M10')
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
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[1]))


def test_model_output_etm(data_dir):
    dataset = Dataset()
    dataset.load(data_dir + '/M10')
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
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[1]))


def test_model_output_ctm_zeroshot(data_dir):
    dataset = Dataset()
    dataset.load(data_dir + '/M10')
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
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[1]))


def test_model_output_ctm_combined(data_dir):
    dataset = Dataset()
    dataset.load(data_dir + '/M10')
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
    assert output['test-topic-document-matrix'].shape == (num_topics, len(dataset.get_partitioned_corpus()[1]))
