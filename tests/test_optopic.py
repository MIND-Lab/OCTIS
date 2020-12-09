#!/usr/bin/env python

"""Tests for `optopic` package."""

import pytest

from click.testing import CliRunner

from optopic.evaluation_metrics.coherence_metrics import *
from optopic.dataset.dataset import Dataset
from optopic.models.LDA import LDA

from optopic import cli

import os


@pytest.fixture
def root_dir():
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def data_dir(root_dir):
    return root_dir + "/../optopic/preprocessed_datasets/"


def test_coherence_measures():
    dataset = Dataset()
    dataset.load(data_dir + '/m10_validation')

    model = LDA(num_topics=3, iterations=5)
    output = model.train_model(dataset)
    metrics_parameters = {'topk': 10, "texts": dataset.get_corpus()}
    metric = Coherence(metrics_parameters)
    score = metric.score(output)
    assert type(score) == np.float64

