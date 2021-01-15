#!/usr/bin/env python

"""Tests for `octis` package."""

import pytest

from click.testing import CliRunner

from octis.evaluation_metrics.coherence_metrics import *
from octis.dataset.dataset import Dataset
from octis.models.LDA import LDA

from octis import cli

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

