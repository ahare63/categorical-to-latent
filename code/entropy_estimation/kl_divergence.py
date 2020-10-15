from collections import Counter
import embeddings
import faiss
import json
from glob import glob
import math
from nltk.tokenize import word_tokenize
import numpy as np
import random
import scipy.special as sc
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import NearestNeighbors
import string

from process_words import get_words, get_dicts, remove_misses
from utils import make_stopwords_list

def get_intersection(a_list, b_list):
    a_new = []
    b_new = []
   
    for a, b in zip(a_list, b_list):
        if a and b:
            a_new.append(a)
            b_new.append(b)
    return np.asarray(a_new), np.asarray(b_new)


# From equation in "Neighbor method for divergence estimation." section
def B(k, alpha):
    return  gammadiv(sc.gammaln(k), sc.gammaln(k -alpha + 1)) * gammadiv(sc.gammaln(k), sc.gammaln(k + alpha - 1))

def gammadiv(a, b):
    return math.exp(a - b)

def first(M, N, rho, upsilon, alpha):
  return math.pow(((N - 1) * rho / (M * upsilon)), 1 - alpha)

# Renamed from "main_f" in "uai.py"
def estimate_KL_divergence(X, Y, k, alphas, gpu=False, distances=None, tree=None, counts=None, M=None):
    # Prevent error where we try to find more neighbors than there are words to look for
    original_k = k
    k = min(k, X.shape[0])
    if original_k != k:
        print("Shrank k from", original_k, "to", k)
    
    if distances is None:
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
        # find the k closest points in X for each point in X
        # Unclear if this excludes the point itself
        distances, _ = nbrs.kneighbors(X)
    if tree is None:
        tree = faiss.IndexFlatL2(Y.shape[1])
        if gpu:
            gpu = faiss.StandardGpuResources()
            tree = faiss.index_cpu_to_gpu(gpu, 0, tree)
        tree.add(Y)

    unique = X.shape[0]
    N = sum(counts)
    if M == None:
        M = Y.shape[0]
    res = 0.0
    if counts is None:
        counts = [1]*N
        
    tmp_query = np.random.rand(1, Y.shape[1]).astype("float32")
    # find the k closest points in Y for each point in X
    # again, unclear if this excludes the point itself
    # 0 gets you the distances rather than the indices they correspond to
    upsilons = tree.search(X, k)[0]

    # reses will hold one divergence calculation for each alpha
    reses = [0] * len(alphas)
    bs = []
    for alpha in alphas:
        bs.append(B(k, alpha))
    for i in range(0, unique):
        rho = distances[i][-1]
        upsilon = upsilons[i][-1]

        # This code appears to prevent a divide by zero error - unclear how or why
        # This code handles finding x as the nearest neighbor?

        # If that's true, then we're only doing k-1 nearest neighbors
        if rho == 0 and upsilon == 0:
            res += 1
            continue
        
        # It looks like this block is trying to avoid the case where there's 
        # a really similar word in Y - possibly to avoid divide by zero?
        # I'm not sure this is kosher though - seems like it would change the result?
        # Why not just "clip" small upsilon by rounding them up to the min value?
        # Changes likely small though
        t = k
        tmp_query[0] = X[i]
        while upsilon < 0.0001 and t <= 900:
            t += 100
            tmp_res = tree.search(tmp_query, t)[0][0]
            for y in tmp_res:
                if y > 0.0001:
                    upsilon = y
                    break
        if upsilon < 0.0001:
            res += 1
            continue

        # Save the results - each one is a single term in the sum
        # When the for loop finishes, reses[l] will be the full sum from 1 to N
        for l in range(0, len(reses)):
            reses[l] += first(M, N, rho, upsilon, alphas[l]) * bs[l] * counts[i]
    # Divide each sum by N, take log, multiply
    for l in range(0, len(reses)):
        reses[l] = (1/(alphas[l] - 1)) * math.log(reses[l]/N, 2)
    return reses


def estimate_JS_divergence(X, Y, k, alphas, gpu=False, distances=None, tree=None, counts=None, M=None):
    M = np.concatenate((X, Y), axis=0)
    div = []
    left = estimate_KL_divergence(X, M, k, alphas, gpu, distances, tree, counts=counts, M=M)
    right = estimate_KL_divergence(Y, M, k, alphas, gpu, distances, tree, counts=counts, M=M)
    for l, r in zip(left, right):
        if 0.5*l + 0.5*r > 1:
            print("ERROR - JS exceeds 1", 0.5*l + 0.5*r)
        div.append(0.5*l + 0.5*r)
    return np.mean(div)

# This gets a mapping from each token to an index so that different sentences can be compared
def get_word_2_ind(word_2_count, stopwords, embedder=None):
    if embedder is None:
        embedder = embeddings.FastTextEmbedding()
    word_2_embedding, _, _, _ = get_dicts(word_2_count, embedder, stopwords)
    word_2_ind = {w:i for i, w in enumerate(word_2_embedding.keys())}
    return word_2_ind

def get_article_dicts(dataset, embedder=None, word_2_ind=None):
    if embedder is None:
        embedder = embeddings.FastTextEmbedding()

    ind_2_auth = {}
    ind_2_text = {}
    ind_2_counts = {}
    ind_2_embs = {}
    ind_2_p = {}
    
    if dataset == 'books_full':
        folders = ["middle", "high", "college"]
        book_2_text = {}
        for folder in folders:
            files = glob(f'./data/books/{folder}/*')
            for text_file in files:
                with open(text_file, "r") as f:
                    t = f.readlines()
                key = (text_file.split("/")[-1]).split(".")[0]
                book_2_text[key] = '\n'.join(t)
    elif dataset.startswith('books_sample'):
        book_2_text = {}
        files = glob(f'./data/{dataset}/*')
        for text_file in files:
            with open(text_file, "r") as f:
                t = f.readlines()
            key = (text_file.split("/")[-1]).split(".")[0]
            book_2_text[key] = '\n'.join(t)

    ind_2_auth = {k:v for k,v in enumerate(book_2_text.keys())}
    auth_2_ind = {k:v for v,k in ind_2_auth.items()}
    ind_2_text = {k:v for v,k in enumerate(book_2_text.values())}

    stopwords = make_stopwords_list()
    # Build index so that each word maps to one index across all books
    text = '\n'.join(list(book_2_text.values()))
    tok = word_tokenize(text)
    word_2_count = dict(Counter(tok).most_common())
    word_2_ind = get_word_2_ind(word_2_count, stopwords)
    vocab_size = len(word_2_ind)

    for auth, text in book_2_text.items():
        ind = auth_2_ind[auth]
        tok = word_tokenize(text)
        word_2_count = dict(Counter(tok).most_common())

        # Get dicts
        word_2_embedding, _, word_2_count, misses = get_dicts(word_2_count, embedder, stopwords)
        word_2_count = remove_misses(word_2_count, misses, word_2_embedding)

        # Retain the embeddings for our method of KL divergence
        ind_2_counts[ind] = word_2_count
        ind_2_embs[ind] = word_2_embedding 
        total_words = float(sum(word_2_count.values()))
        for word, count in word_2_count.items():
            if word not in word_2_ind:
                print("word", word, "count", count)
                total_words -= count
        # Calculate p for each article
        p = [0]*vocab_size
        for word, count in word_2_count.items():
            if word in word_2_ind:
                p[word_2_ind[word]] = count/total_words
        ind_2_p[ind] = p
        
    return ind_2_auth, ind_2_text, ind_2_counts, ind_2_embs, ind_2_p


def save_distances(ind_2_auth, ind_2_embs, k=[3, 5, 10, 25, 50, 100]):
    # Get distances for all authors
    auth_2_dis = {}
    for ind, auth in ind_2_auth.items():
        X = np.asarray(list(ind_2_embs[ind].values()))
        auth_2_dis[auth] = {}
        auth_2_dis[auth]["X"] = X
        for key in k:
            nbrs = NearestNeighbors(n_neighbors=key, algorithm='kd_tree').fit(X)
            distances, _ = nbrs.kneighbors(X)
            auth_2_dis[auth][key] = distances
    # Write results to file
    with open('./kl_results/distances.json', 'w') as f:
        json.dump(auth_2_dis, f, indent=2)


def get_kl_divs(dataset, ind_2_p, ind_2_auth, ind_2_embs, ind_2_counts, k=[3, 5, 10, 25, 50, 100], alphas=[0.99, 1.01], gpu=False, jsd=False, efficient=False, start_ind=0, target_list=None):
    auth_2_dis = {}
    for ind_l, auth in ind_2_auth.items():
        # Try only books after the start index
        if ind_l < start_ind:
            continue
        # Try only books in target_list
        if target_list is not None and auth not in target_list:
            continue
        # Print the book we're processing
        print(auth)
        l_emb = np.asarray(list(ind_2_embs[ind_l].values()))
        counts = []
        for ind in ind_2_embs[ind_l].keys():
            counts.append(ind_2_counts[ind_l][ind])

        # Save the nearest neighbors to avoid re-computation
        X = l_emb
        auth_2_dis[auth] = {}
        auth_2_dis[auth]["X"] = X
        for key in k:
            nbrs = NearestNeighbors(n_neighbors=key, algorithm='kd_tree').fit(X)
            distances, _ = nbrs.kneighbors(X)
            auth_2_dis[auth][key] = distances
        
        comps = {}  # The comparisons we're saving
        # Iterate through all other books
        for ind_r, auth_r in ind_2_auth.items():
            # Skip the same book
            if ind_l == ind_r:
                continue
            r_emb = np.asarray(list(ind_2_embs[ind_r].values()))
            right_counts = []
            for ind in ind_2_embs[ind_r].keys():
                right_counts.append(ind_2_counts[ind_r][ind])
            M = sum(right_counts)
            results = {}
            if not jsd:
                x, y = get_intersection(ind_2_p[ind_l], ind_2_p[ind_r])
                og_kl = entropy(x, qk=y, base=2)
                if math.isinf(og_kl):
                    print("Same is infinity")
                
                for key in k:
                    results[key] = np.mean(estimate_KL_divergence(l_emb, r_emb, key, alphas, gpu, distances=auth_2_dis[auth][key], counts=counts, M=M))
            else:
                og_kl = jensenshannon(ind_2_p[ind_l], ind_2_p[ind_r], base=2)
                kl_dict = {}
                for key in k:
                    kl_dict[key] = np.mean(estimate_JS_divergence(l_emb, r_emb, key, alphas, gpu, distances=auth_2_dis[auth][key], counts=counts, M=M))

            results["original"] = og_kl
            comps[auth_r] = results
        # Write results to file
        with open(f'./kl_results/{dataset}/{auth}.json', 'w') as f:
            json.dump(comps, f, indent=2)
