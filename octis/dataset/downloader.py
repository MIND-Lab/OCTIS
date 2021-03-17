from os import environ, makedirs
from os.path import exists, expanduser, join, splitext
import pickle
import sys
import codecs
import shutil
import requests

"""
This code is highly inspired by the scikit-learn strategy to download datasets
"""


def get_data_home(data_home=None):
    """Return the path of the octis data dir.
    By default the data dir is set to a folder named 'octis_data' in the
    user home folder.
    Alternatively, it can be set by the 'OCTIS_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.
    Parameters
    ----------
    data_home : str | None
        The path to octis data dir.
    """
    if data_home is None:
        data_home = environ.get('OCTIS_DATA', join('~', 'octis_data'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def _pkl_filepath(*args, **kwargs):
    """Ensure different filenames for Python 2 and Python 3 pickles
    An object pickled under Python 3 cannot be loaded under Python 2. An object
    pickled under Python 2 can sometimes not be loaded correctly under Python 3
    because some Python 2 strings are decoded as Python 3 strings which can be
    problematic for objects that use Python 2 strings as byte buffers for
    numerical data instead of "real" strings.
    Therefore, dataset loaders in octis use different files for pickles
    manages by Python 2 and Python 3 in the same OCTIS_DATA folder so as
    to avoid conflicts.
    args[-1] is expected to be the ".pkl" filename. Under Python 3, a suffix is
    inserted before the extension to s
    _pkl_filepath('/path/to/folder', 'filename.pkl') returns:
      - /path/to/folder/filename.pkl under Python 2
      - /path/to/folder/filename_py3.pkl under Python 3+
    """
    py3_suffix = kwargs.get("py3_suffix", "_py3")
    basename, ext = splitext(args[-1])
    if sys.version_info[0] >= 3:
        basename += py3_suffix
    new_args = args[:-1] + (basename + ext,)
    return join(*new_args)


def download_dataset(dataset_name, target_dir, cache_path):
    """Download the 20 newsgroups data and stored it as a zipped pickle."""
    corpus_path = join(target_dir, "corpus.txt")
    label_path = join(target_dir, "labels.txt")
    metadata_path = join(target_dir, "metadata.json")
    vocabulary_path = join(target_dir, "vocabulary.txt")

    if not exists(target_dir):
        makedirs(target_dir)

    dataset_url = "https://raw.githubusercontent.com/MIND-Lab/OCTIS/master/octis/preprocessed_datasets/" + dataset_name

    corpus = requests.get(dataset_url + "/corpus.txt")
    labels = requests.get(dataset_url + "/labels.txt")
    metadata = requests.get(dataset_url + "/metadata.json")
    vocabulary = requests.get(dataset_url + "/vocabulary.txt")

    if corpus and labels and metadata and vocabulary:
        with open(label_path, 'w') as f:
            f.write(labels.text)
        with open(corpus_path, 'w') as f:
            f.write(corpus.text)
        with open(metadata_path, 'w') as f:
            f.write(metadata.text)
        with open(vocabulary_path, 'w') as f:
            f.write(vocabulary.text)

        # Store a zipped pickle
        cache = dict(corpus=corpus.text, labels=labels.text, metadata=metadata.text, vocabulary=vocabulary.text)
        compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
        with open(cache_path, 'wb') as f:
            f.write(compressed_content)

        shutil.rmtree(target_dir)
        return cache
    else:
        raise Exception(dataset_name + ' dataset not found')


