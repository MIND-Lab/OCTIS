import os
import string
from octis.preprocessing.preprocessing import Preprocessing
os.chdir(os.path.pardir)

p = Preprocessing(vocabulary=None, max_features=None, remove_punctuation=True, punctuation=string.punctuation,
                  lemmatize=True, remove_stopwords=True, stopword_list=['am', 'are', 'this', 'that'],
                  min_chars=1, min_words_docs=0)
dataset = p.preprocess_dataset(
    documents_path=r'..\preprocessed_datasets\M10\corpus.txt',
    labels_path=r'..\preprocessed_datasets\M10\labels.txt',
)

dataset.save('hello_dataset.txt')
