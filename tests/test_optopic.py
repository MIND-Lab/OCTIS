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
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'optopic.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


def test_coherence_measures():

    # get current directory
    path = os.getcwd()
    # prints parent directory
    parent_path = str(os.path.abspath(os.path.join(path, os.pardir)))

    dataset = Dataset()
    dataset.load(parent_path + '/optopic/preprocessed_datasets/m10_validation')
    print(os.path.exists(parent_path + '/optopic/preprocessed_datasets/m10_validation'))
    print(os.path.exists(parent_path + '/optopic/'))
    print(os.path.exists(parent_path))
    print(parent_path)
    print(path)

    model = LDA(num_topics=3, iterations=5)
    output = model.train_model(dataset)
    metrics_parameters = {'topk': 10, "texts": dataset.get_corpus()}
    metric = Coherence(metrics_parameters)
    score = metric.score(output)
    assert type(score) == np.float64

