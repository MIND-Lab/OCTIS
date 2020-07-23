from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle
import random
from scipy import sparse
import itertools
from scipy.io import savemat, loadmat

# Maximum / minimum document frequency
max_df = 0.7
min_df = 100  # choose desired value for min_df

# Read stopwords
with open('stops.txt', 'r') as f:
    stops = f.read().split('\n')

# Read data
print('reading text file...')
data_file = 'raw/new_york_times_text/nyt_docs.txt'
with open(data_file, 'r') as f:
    docs = f.readlines()

# Create count vectorizer
print('counting document frequency of words...')
cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=None)
cvz = cvectorizer.fit_transform(docs).sign()

# Get vocabulary
print('building the vocabulary...')
sum_counts = cvz.sum(axis=0)
v_size = sum_counts.shape[1]
sum_counts_np = np.zeros(v_size, dtype=int)
for v in range(v_size):
    sum_counts_np[v] = sum_counts[0,v]
word2id = dict([(w, cvectorizer.vocabulary_.get(w)) for w in cvectorizer.vocabulary_])
id2word = dict([(cvectorizer.vocabulary_.get(w), w) for w in cvectorizer.vocabulary_])
del cvectorizer
print('  initial vocabulary size: {}'.format(v_size))

# Sort elements in vocabulary
idx_sort = np.argsort(sum_counts_np)
vocab_aux = [id2word[idx_sort[cc]] for cc in range(v_size)]

# Filter out stopwords (if any)
vocab_aux = [w for w in vocab_aux if w not in stops]
print('  vocabulary size after removing stopwords from list: {}'.format(len(vocab_aux)))
print('  vocabulary after removing stopwords: {}'.format(len(vocab_aux)))

# Create dictionary and inverse dictionary
vocab = vocab_aux
del vocab_aux
word2id = dict([(w, j) for j, w in enumerate(vocab)])
id2word = dict([(j, w) for j, w in enumerate(vocab)])

# Split in train/test/valid
print('tokenizing documents and splitting into train/test/valid...')
num_docs = cvz.shape[0]
trSize = int(np.floor(0.85*num_docs))
tsSize = int(np.floor(0.10*num_docs))
vaSize = int(num_docs - trSize - tsSize)
del cvz
idx_permute = np.random.permutation(num_docs).astype(int)

# Remove words not in train_data
vocab = list(set([w for idx_d in range(trSize) for w in docs[idx_permute[idx_d]].split() if w in word2id]))
word2id = dict([(w, j) for j, w in enumerate(vocab)])
id2word = dict([(j, w) for j, w in enumerate(vocab)])
print('  vocabulary after removing words not in train: {}'.format(len(vocab)))

docs_tr = [[word2id[w] for w in docs[idx_permute[idx_d]].split() if w in word2id] for idx_d in range(trSize)]
docs_ts = [[word2id[w] for w in docs[idx_permute[idx_d+trSize]].split() if w in word2id] for idx_d in range(tsSize)]
docs_va = [[word2id[w] for w in docs[idx_permute[idx_d+trSize+tsSize]].split() if w in word2id] for idx_d in range(vaSize)]
del docs

print('  number of documents (train): {} [this should be equal to {}]'.format(len(docs_tr), trSize))
print('  number of documents (test): {} [this should be equal to {}]'.format(len(docs_ts), tsSize))
print('  number of documents (valid): {} [this should be equal to {}]'.format(len(docs_va), vaSize))

# Remove empty documents
print('removing empty documents...')

def remove_empty(in_docs):
    return [doc for doc in in_docs if doc!=[]]

docs_tr = remove_empty(docs_tr)
docs_ts = remove_empty(docs_ts)
docs_va = remove_empty(docs_va)

# Remove test documents with length=1
docs_ts = [doc for doc in docs_ts if len(doc)>1]

# Split test set in 2 halves
print('splitting test documents in 2 halves...')
docs_ts_h1 = [[w for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc in docs_ts]
docs_ts_h2 = [[w for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc in docs_ts]

# Getting lists of words and doc_indices
print('creating lists of words...')

def create_list_words(in_docs):
    return [x for y in in_docs for x in y]

words_tr = create_list_words(docs_tr)
words_ts = create_list_words(docs_ts)
words_ts_h1 = create_list_words(docs_ts_h1)
words_ts_h2 = create_list_words(docs_ts_h2)
words_va = create_list_words(docs_va)

print('  len(words_tr): ', len(words_tr))
print('  len(words_ts): ', len(words_ts))
print('  len(words_ts_h1): ', len(words_ts_h1))
print('  len(words_ts_h2): ', len(words_ts_h2))
print('  len(words_va): ', len(words_va))

# Get doc indices
print('getting doc indices...')

def create_doc_indices(in_docs):
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(x) for y in aux for x in y]

doc_indices_tr = create_doc_indices(docs_tr)
doc_indices_ts = create_doc_indices(docs_ts)
doc_indices_ts_h1 = create_doc_indices(docs_ts_h1)
doc_indices_ts_h2 = create_doc_indices(docs_ts_h2)
doc_indices_va = create_doc_indices(docs_va)

print('  len(np.unique(doc_indices_tr)): {} [this should be {}]'.format(len(np.unique(doc_indices_tr)), len(docs_tr)))
print('  len(np.unique(doc_indices_ts)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts)), len(docs_ts)))
print('  len(np.unique(doc_indices_ts_h1)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts_h1)), len(docs_ts_h1)))
print('  len(np.unique(doc_indices_ts_h2)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts_h2)), len(docs_ts_h2)))
print('  len(np.unique(doc_indices_va)): {} [this should be {}]'.format(len(np.unique(doc_indices_va)), len(docs_va)))

# Number of documents in each set
n_docs_tr = len(docs_tr)
n_docs_ts = len(docs_ts)
n_docs_ts_h1 = len(docs_ts_h1)
n_docs_ts_h2 = len(docs_ts_h2)
n_docs_va = len(docs_va)

# Remove unused variables
del docs_tr
del docs_ts
del docs_ts_h1
del docs_ts_h2
del docs_va

# Create bow representation
print('creating bow representation...')

def create_bow(doc_indices, words, n_docs, vocab_size):
    return sparse.coo_matrix(([1]*len(doc_indices),(doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()

bow_tr = create_bow(doc_indices_tr, words_tr, n_docs_tr, len(vocab))
bow_ts = create_bow(doc_indices_ts, words_ts, n_docs_ts, len(vocab))
bow_ts_h1 = create_bow(doc_indices_ts_h1, words_ts_h1, n_docs_ts_h1, len(vocab))
bow_ts_h2 = create_bow(doc_indices_ts_h2, words_ts_h2, n_docs_ts_h2, len(vocab))
bow_va = create_bow(doc_indices_va, words_va, n_docs_va, len(vocab))

del words_tr
del words_ts
del words_ts_h1
del words_ts_h2
del words_va
del doc_indices_tr
del doc_indices_ts
del doc_indices_ts_h1
del doc_indices_ts_h2
del doc_indices_va

# Save vocabulary to file
path_save = './min_df_' + str(min_df) + '/'
if not os.path.isdir(path_save):
    os.system('mkdir -p ' + path_save)

with open(path_save + 'vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
del vocab

# Split bow intro token/value pairs
print('splitting bow intro token/value pairs and saving to disk...')

def split_bow(bow_in, n_docs):
    indices = [[w for w in bow_in[doc,:].indices] for doc in range(n_docs)]
    counts = [[c for c in bow_in[doc,:].data] for doc in range(n_docs)]
    return indices, counts

bow_tr_tokens, bow_tr_counts = split_bow(bow_tr, n_docs_tr)
savemat(path_save + 'bow_tr_tokens', {'tokens': bow_tr_tokens}, do_compression=True)
savemat(path_save + 'bow_tr_counts', {'counts': bow_tr_counts}, do_compression=True)
del bow_tr
del bow_tr_tokens
del bow_tr_counts

bow_ts_tokens, bow_ts_counts = split_bow(bow_ts, n_docs_ts)
savemat(path_save + 'bow_ts_tokens', {'tokens': bow_ts_tokens}, do_compression=True)
savemat(path_save + 'bow_ts_counts', {'counts': bow_ts_counts}, do_compression=True)
del bow_ts
del bow_ts_tokens
del bow_ts_counts

bow_ts_h1_tokens, bow_ts_h1_counts = split_bow(bow_ts_h1, n_docs_ts_h1)
savemat(path_save + 'bow_ts_h1_tokens', {'tokens': bow_ts_h1_tokens}, do_compression=True)
savemat(path_save + 'bow_ts_h1_counts', {'counts': bow_ts_h1_counts}, do_compression=True)
del bow_ts_h1
del bow_ts_h1_tokens
del bow_ts_h1_counts

bow_ts_h2_tokens, bow_ts_h2_counts = split_bow(bow_ts_h2, n_docs_ts_h2)
savemat(path_save + 'bow_ts_h2_tokens', {'tokens': bow_ts_h2_tokens}, do_compression=True)
savemat(path_save + 'bow_ts_h2_counts', {'counts': bow_ts_h2_counts}, do_compression=True)
del bow_ts_h2
del bow_ts_h2_tokens
del bow_ts_h2_counts

bow_va_tokens, bow_va_counts = split_bow(bow_va, n_docs_va)
savemat(path_save + 'bow_va_tokens', {'tokens': bow_va_tokens}, do_compression=True)
savemat(path_save + 'bow_va_counts', {'counts': bow_va_counts}, do_compression=True)
del bow_va
del bow_va_tokens
del bow_va_counts

print('Data ready !!')
print('*************')

