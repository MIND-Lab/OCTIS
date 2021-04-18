#!/usr/bin/env python

"""Tests for `octis` package."""

import pytest

from click.testing import CliRunner
from octis.evaluation_metrics.topic_significance_metrics import *
from octis.evaluation_metrics.classification_metrics import F1Score, PrecisionScore
from octis.evaluation_metrics.classification_metrics import  AccuracyScore, RecallScore
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO

from octis.evaluation_metrics.coherence_metrics import *
from octis.dataset.dataset import Dataset
from octis.models.LDA import LDA

import os


@pytest.fixture
def root_dir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def dataset(root_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(root_dir + "/../preprocessed_datasets/" + '/M10')
    return dataset


@pytest.fixture
def model_output(dataset):
    model = LDA(num_topics=3, iterations=5)
    output = model.train_model(dataset)
    return output


def test_f1score(dataset, model_output):
    metric = F1Score(dataset=dataset)
    score = metric.score(model_output)
    assert type(score) == np.float64 or type(score) == float


def test_accuracyscore(dataset, model_output):
    metric = AccuracyScore(dataset=dataset)
    score = metric.score(model_output)
    assert type(score) == np.float64 or type(score) == float


def test_precisionscore(dataset, model_output):
    metric = PrecisionScore(dataset=dataset)
    score = metric.score(model_output)
    assert type(score) == np.float64 or type(score) == float


def test_recallscore(dataset, model_output):
    metric = RecallScore(dataset=dataset)
    score = metric.score(model_output)
    assert type(score) == np.float64 or type(score) == float


def test_svm_persistency(dataset, model_output):
    metric = F1Score(dataset=dataset)
    metric.score(model_output)
    metric = AccuracyScore(dataset=dataset)
    metric.score(model_output)
    assert metric.same_svm
    metric = F1Score(dataset=dataset, average="macro")
    metric.score(model_output)
    assert not metric.same_svm


def test_npmi_coherence_measures(dataset, model_output):
    metric = Coherence(topk=10, texts=dataset.get_corpus())
    score = metric.score(model_output)
    assert type(score) == np.float64 or type(score) == float


def test_we_coherence_measures(dataset, model_output):
    metric = WECoherenceCentroid(topk=5)
    score = metric.score(model_output)
    assert type(score) == np.float64 or type(score) == np.float32 or type(score) == float

    metric = WECoherencePairwise(topk=10)
    score = metric.score(model_output)
    assert type(score) == np.float64 or type(score) == np.float32 or type(score) == float


def test_diversity_measures(dataset, model_output):
    metric = TopicDiversity(topk=10)
    score = metric.score(model_output)
    assert type(score) == np.float64 or type(score) == float


def test_irbo(dataset, model_output):
    metric = InvertedRBO(topk=10)
    score = metric.score(model_output)
    assert type(score) == np.float64 or type(score) == float


def test_kl_b(dataset, model_output):
    metric = KL_background()
    score = metric.score(model_output)
    assert type(score) == np.float64 or type(score) == float


def test_kl_v(dataset, model_output):
    metric = KL_vacuous()
    score = metric.score(model_output)
    assert type(score) == np.float64 or type(score) == float


def test_kl_u(dataset, model_output):
    metric = KL_uniform()
    score = metric.score(model_output)
    assert type(score) == np.float64 or type(score) == float
