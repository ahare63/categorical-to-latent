import faiss

def get_index(emb_mapping, use_gpu=False):
  key_2_ind = {k: v for v, k in enumerate(emb_mapping.keys())}
  ind_2_key = {k: v for v, k in key_2_ind.items()}
    
  data_array = np.asarray(list(emb_mapping.values())).astype('float32')
  faiss_index = faiss.IndexFlatL2(data_array.shape[1])

  if use_gpu:
    gpu = faiss.StandardGpuResources()
    faiss_index = faiss.index_cpu_to_gpu(gpu, 0, faiss_index)

  faiss_index.add(data_array)
  return faiss_index, key_2_ind, ind_2_key


from gensim.corpora.dictionary import Dictionary
from pyemd import emd
import numpy as np
from math import sqrt

def wmdistance(document1, document2, embedder, distance_matrix=None):
    """
    Compute the Word Mover's Distance between two documents. When using this
    code, please consider citing the following papers:

    .. Ofir Pele and Michael Werman, "A linear time histogram metric for improved SIFT matching".
    .. Ofir Pele and Michael Werman, "Fast and robust earth mover's distances".
    .. Matt Kusner et al. "From Word Embeddings To Document Distances".
"""

    dictionary = Dictionary(documents=[document1, document2])
    vocab_len = len(dictionary)

    # Compute distance matrix if not provided. This is the "brute-force" solution.
    # Sets for faster look-up.
    docset1 = set(document1)
    docset2 = set(document2)
    if distance_matrix is None:
        distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
        for i, t1 in dictionary.items():
            for j, t2 in dictionary.items():
                if not t1 in docset1 or not t2 in docset2:
                    continue
                # Compute Euclidean distance between word vectors.
                distance_matrix[i, j] = sqrt(np.sum((np.asarray(embedder.emb(t1)) - np.asarray(embedder.emb(t2)))**2))


    if np.sum(distance_matrix) == 0.0:
        return float('inf')

    def nbow(document):
        d = np.zeros(vocab_len, dtype=np.double)
        nbow = dictionary.doc2bow(document)  # Word frequencies.
        doc_len = len(document)
        for idx, freq in nbow:
            d[idx] = freq / float(doc_len)  # Normalized word frequencies.
        return d

    # Compute nBOW representation of documents.
    d1 = nbow(document1)
    d2 = nbow(document2)

    # Compute WMD.
    return emd(d1, d2, distance_matrix)


from nltk.corpus import stopwords as nltk_stopwords
def make_stopwords_list():
  stopwords = list(nltk_stopwords.words('english'))
  # Add capitalized version of stopwords. str.title() messes up contractions
  stopwords += list([x[0].upper() + x[1:] for x in stopwords])
  # Add punctuation
  stopwords += [".", ",", "!", "?", "...", "-", ":", ";", "“"]
  return stopwords

def jaccard_similarity(A, B, debug=False):
    A = set(A)
    B = set(B)
    intersection = len(A.intersection(B))
    if debug:
        print("Intersection", intersection)
    union = len(A) + len(B) - intersection
    return intersection/union