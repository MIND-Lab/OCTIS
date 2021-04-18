#!/usr/bin/env python

"""Tests for `octis` package."""

import pytest

from click.testing import CliRunner
from octis.evaluation_metrics.classification_metrics import F1Score

from octis.evaluation_metrics.coherence_metrics import *
from octis.dataset.dataset import Dataset
from octis.models.LDA import LDA
from octis.models.ETM import ETM
from octis.models.CTM import CTM
from octis.models.NMF import NMF
from octis.models.NMF_scikit import NMF_scikit
from octis.models.ProdLDA import ProdLDA

import os
from octis.preprocessing.preprocessing import Preprocessing

from octis.dataset.downloader import get_data_home, _pkl_filepath

@pytest.fixture
def root_dir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def data_dir(root_dir):
    return root_dir + "/../preprocessed_datasets/"


def test_preprocessing(data_dir):
    texts_path = data_dir+"/sample_texts/unprepr_docs.txt"
    p = Preprocessing(vocabulary=None, max_features=None, remove_punctuation=True, punctuation=".,?:",
                      lemmatize=False,  stopword_list=['am', 'are', 'this', 'that'],
                      min_chars=2, min_words_docs=5,min_df=0.0001)
    dataset = p.preprocess_dataset(
        documents_path=texts_path,
    )

    dataset.save(data_dir+"/sample_texts/")
    dataset.load_custom_dataset_from_folder(data_dir + "/sample_texts")


def test_load_20ng():
    data_home = get_data_home(data_home=None)
    cache_path = _pkl_filepath(data_home, "20NewsGroup" + ".pkz")
    if os.path.exists(cache_path):
        os.remove(cache_path)

    dataset = Dataset()
    dataset.fetch_dataset("20NewsGroup")
    assert len(dataset.get_corpus()) == 16309
    assert len(dataset.get_labels()) == 16309
    assert os.path.exists(cache_path)

    dataset = Dataset()
    dataset.fetch_dataset("20NewsGroup")
    assert len(dataset.get_corpus()) == 16309


def test_load_M10():
    dataset = Dataset()
    dataset.fetch_dataset("M10")
    assert len(set(dataset.get_labels())) == 10


def test_partitions_fetch():
    dataset = Dataset()
    dataset.fetch_dataset("M10")
    partitions = dataset.get_partitioned_corpus()
    assert len(partitions[0]) == 5847
    assert len(partitions[1]) == 1254


def test_partitions_custom(data_dir):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_dir+"M10")
    partitions = dataset.get_partitioned_corpus()
    assert len(partitions[0]) == 5847
    assert len(partitions[1]) == 1254
