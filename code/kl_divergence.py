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
def estimate_KL_divergence(X, Y, k, alphas, gpu=False, distances=None, tree=None):
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

    N = X.shape[0]
    M = Y.shape[0]
    res = 0.0
        
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
    for i in range(0, N):
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
            reses[l] += first(M, N, rho, upsilon, alphas[l]) * bs[l]
    # Divide each sum by N, take log, multiply
    for l in range(0, len(reses)):
        reses[l] = (1/(alphas[l] - 1)) * math.log(reses[l]/N, 2)
    return reses


def estimate_JS_divergence(X, Y, k, alphas, gpu=False, distances=None, tree=None):
    M = np.concatenate((X, Y), axis=0)
    div = []
    left = estimate_KL_divergence(X, M, k, alphas, gpu, distances, tree)
    right = estimate_KL_divergence(Y, M, k, alphas, gpu, distances, tree)
    for l, r in zip(left, right):
        if 0.5*l + 0.5*r > 1:
            print("ERROR - JS exceeds 1", 0.5*l + 0.5*r)
        div.append(0.5*l + 0.5*r)
    return np.mean(div)

# This gets a mapping from each token to an index so that different sentences can be compared
def get_word_2_ind(dataset, embedder=None):
    if embedder is None:
        embedder = embeddings.FastTextEmbedding()
    word_2_embedding, _, _, _ = get_words(dataset, embedder, keep_misses=False)
    word_2_ind = {w:i for i, w in enumerate(word_2_embedding.keys())}
    return word_2_ind

def get_article_dicts(dataset, embedder=None, word_2_ind=None):
    if embedder is None:
        embedder = embeddings.FastTextEmbedding()
    if word_2_ind is None:
        word_2_ind = get_word_2_ind(dataset, embedder)

    ind_2_auth = {}
    ind_2_text = {}
    ind_2_counts = {}
    ind_2_embs = {}
    ind_2_p = {}
    vocab_size = len(word_2_ind)

    if dataset == 'authorship':
        with open('./data/authorship_data.json', 'rb') as f:
            j = json.load(f)

            for key in j.keys():
                ind = key
                ind_2_auth[ind] = j[key]['author']
                ind_2_text[ind] = j[key]['text']
                
                # Tokenize words
                text = word_tokenize(j[key]['text'])
                word_2_count = dict(Counter(text).most_common())
                # Get dicts
                word_2_embedding, _, word_2_count, misses = get_dicts(word_2_count, embedder, [])
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
                    p[word_2_ind[word]] = count/total_words
                ind_2_p[ind] = p
    
    elif dataset == 'books':
        folders = ["middle", "high", "college"]
        book_2_text = {}
        for folder in folders:
            files = glob(f'./data/books/{folder}/*')
            for file in files:
                with open(file, "r") as f:
                    t = f.readlines()
                key = (file.split("/")[-1]).split(".")[0]
                book_2_text[key] = '\n'.join(t)
        ind_2_auth = {k:v for k,v in enumerate(book_2_text.keys())}
        auth_2_ind = {k:v for v,k in ind_2_auth.items()}
        ind_2_text = {k:v for v,k in enumerate(book_2_text.values())}
        for auth, text in book_2_text.items():
            ind = auth_2_ind[auth]
            tok = word_tokenize(text)
            word_2_count = dict(Counter(tok).most_common())
            # Get dicts
            word_2_embedding, _, word_2_count, misses = get_dicts(word_2_count, embedder, [])
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


def get_kl_divs(ind_2_p, ind_2_auth, ind_2_embs, k=[3, 5, 10, 25, 50, 100], alphas=[0.99, 1.01], gpu=False, jsd=False, efficient=False, start_ind=0, target_list=None):
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
            results = {}
            if not jsd:
                x, y = get_intersection(ind_2_p[ind_l], ind_2_p[ind_r])
                og_kl = entropy(x, qk=y, base=2)
                if math.isinf(og_kl):
                    print("Same is infinity")
                
                for key in k:
                    results[key] = np.mean(estimate_KL_divergence(l_emb, r_emb, key, alphas, gpu, distances=auth_2_dis[auth][key]))
            else:
                og_kl = jensenshannon(ind_2_p[ind_l], ind_2_p[ind_r], base=2)
                kl_dict = {}
                for key in k:
                    kl_dict[key] = np.mean(estimate_JS_divergence(l_emb, r_emb, key, alphas, gpu, distances=auth_2_dis[auth][key]))

            results["original"] = og_kl
            comps[auth_r] = results
        # Write results to file
        with open(f'./kl_results/{auth}.json', 'w') as f:
            json.dump(comps, f, indent=2)

        # if efficient:
    #     # This approach first initializes the distances for all nearest neighbors
    #     # Then it iterates through 
    #     for ind_r, auth_r in ind_2_auth.items():
    #         # Start at a specified index
    #         if ind_r < start_ind:
    #             continue
    #         # Try only books in target_list
    #         if target_list is not None and auth_r not in target_list:
    #             continue
    #         print(auth_r)
            
    #         r_emb = np.asarray(list(ind_2_embs[ind_r].values()))
    #         # Initialize tree which will be used for each query
    #         tree = faiss.IndexFlatL2(r_emb.shape[1])
    #         if gpu:
    #             gpu = faiss.StandardGpuResources()
    #             tree = faiss.index_cpu_to_gpu(gpu, 0, tree)
    #         tree.add(r_emb)
    #         comps = {}

    #         for ind_l, auth_l in ind_2_auth.items():
    #             if ind_l == ind_r:
    #                 continue
    #             results = {}
    #             if not jsd:
    #                 x, y = get_intersection(ind_2_p[ind_l], ind_2_p[ind_r])
    #                 og_kl = entropy(x, qk=y, base=2)
    #                 if math.isinf(og_kl):
    #                     print("Same is infinity")
                    
    #                 for key in k:
    #                     results[key] = np.mean(estimate_KL_divergence(auth_2_dis[auth_l]["X"], r_emb, key, alphas, gpu, nbrs=auth_2_dis[auth_l][k], tree=tree))
    #             else:
    #                 og_kl = jensenshannon(ind_2_p[ind_l], ind_2_p[ind_r], base=2)
    #                 kl_dict = {}
    #                 for key in k:
    #                     kl_dict[key] = np.mean(estimate_JS_divergence(auth_2_dis[auth_l]["X"], r_emb, key, alphas, gpu, nbrs=auth_2_dis[auth_l][k], tree=tree))

    #             results["original"] = og_kl
    #             comps[auth_r] = results
    #         # Write results to file
    #         with open(f'./kl_results/{auth_r}_r.json', 'w') as f:
    #             json.dump(comps, f, indent=2)


# def get_kl_divs(ind_2_p, ind_2_auth, ind_2_embs, n_comp=99, k=[3, 5, 10, 25, 50, 100], alphas=[0.96, 0.97, 0.98, 0.99], gpu=False, jsd=False):
#     same = []
#     dif = []

#     for i in ind_2_auth.keys():
#         i_emb = np.asarray(list(ind_2_embs[i].values()))
#         # Get ids w/same author
#         same_inds = [k for k, v in ind_2_auth.items() if v == ind_2_auth[i] and k != i]
#         random.shuffle(same_inds)
#         for j in same_inds[:n_comp]:
#             j_emb = np.asarray(list(ind_2_embs[j].values()))
#             if not jsd:
#                 x, y = get_intersection(ind_2_p[i], ind_2_p[j])
#                 og_kl = entropy(x, qk=y, base=2)
#                 if math.isinf(og_kl):
#                     print("Same is infinity")
#                 kl_dict = {}
#                 for key in k:
#                     kl_dict[key] = np.mean(estimate_KL_divergence(i_emb, j_emb, key, alphas, gpu))
#             else:
#                 og_kl = jensenshannon(ind_2_p[i], ind_2_p[j], base=2)
#                 kl_dict = {}
#                 for key in k:
#                     kl_dict[key] = np.mean(estimate_JS_divergence(i_emb, j_emb, key, alphas, gpu))
#             same.append((og_kl, kl_dict))
        
#         # Get random ids w/different author
#         all_inds = list(ind_2_auth.keys())
#         # Remove ids with the same author
#         for ind in same_inds:
#             all_inds.remove(ind)
#         all_inds.remove(i)

#         random.shuffle(all_inds)
        
#         for j in all_inds[:n_comp]:
#             j_emb = np.asarray(list(ind_2_embs[j].values()))
#             if not jsd:
#                 x, y = get_intersection(ind_2_p[i], ind_2_p[j])
#                 og_kl = entropy(x, qk=y, base=2)
#                 if math.isinf(og_kl):
#                     print("Dif is infinity")
#                 kl_dict = {}
#                 for key in k:
#                     kl_dict[key] = np.mean(estimate_KL_divergence(i_emb, j_emb, key, alphas, gpu))
#             else:
#                 og_kl = jensenshannon(ind_2_p[i], ind_2_p[j], base=2)
#                 kl_dict = {}
#                 for key in k:
#                     kl_dict[key] = np.mean(estimate_JS_divergence(i_emb, j_emb, key, alphas, gpu))
#             dif.append((og_kl, kl_dict))
                     
            
#     return same, dif
                