from collections import Counter
import faiss
from gensim.corpora.dictionary import Dictionary
import Levenshtein
from math import sqrt
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

import process_sentences
import process_words
import utils


class SimilarSentences():
  def __init__(self, data_name, embedder):
    self.data_name = data_name      # The dataset to use
    self.embedder = embedder        # The embedder object
    self.stopwords = []             # A list of stopwords
    self.r = 10                      # The number of nearest embedded words to use for set cover
    self.k = 5                      # The number of nearest neighbor sentences to return
    self.length_penalty = 0.5       # The length penalty for the set cover 
    self.use_wt = False             # Use the (TF-IDF) weights as part of the set cover calculation
    self.use_dis = False            # Use the (Euclidean) distance between this and the original word to up-weight closer matches
    self.tfidf = None               # TF-IDF model, used with calc_wt
    self.wmd_k = 10                 # Number of non-zero values to use for wmd estimation
    self.wmd_default = np.inf       # Value to use for all neighbors > wmd_kth closest
    self.use_wmd_estimate = False   # If True, use the wmd estimate. Otherwise, use original
    self.use_wmd_memory = True      # If True, use wmd with "memory". Otherwise, use original.
    self.wmd_memory = {}            # Dict to store Euclidian distances for future lookups
    self.wmd_neighbors = None       # The neighbors of the most recent query, used for approximate WMD
    self.wmd_prev_query = None      # The query corresponding to wmd_neighbors
    self.debug = False              # If True, print information
    self.prevent_duplicate = True   # If True, prevent results from returning something very close to the query sentence

    self.sen_index = None       # faiss index containing average embeddings for a sentence
    self.sen_2_ind = None       # Mapping from string sentence to its number in the index
    self.ind_2_sen = None       # Mapping from index to string representation
    self.ind_2_tok_sen = None   # Mapping from index to tokenized list (all items in list can be embedded)
    self.sen_array = None       # Array of representations of each sentence (input to index)

    self.word_index = None      # faiss index containing embeddings for each word
    self.word_2_ind = None      # Mapping from string word to its number in the index
    self.ind_2_word = None      # Mapping from index to string representation
    self.word_array = None      # Array of representations of each word (input to index)
    self.word_2_count = None    # Mapping of string word to number of appearances in data set
    self.word_2_pos = None      # Mapping of string word to its part of speech

    self.sen_2_emb = None       # Mapping of sentence to its average embedding
    self.word_2_emb = None      # Mapping of word to its embedding
    
  """ Functions for updating parameters """
  # Update search parameters, especially those used for set cover
  def update_search_params(self, r=None, k=None, length_penalty=None, use_wt=None, 
                            use_dis=None, wmd_k=None, wmd_default=None, use_wmd_estimate=None, 
                            use_wmd_memory=None, debug=None, prevent_duplicate=None):
    if r is not None:
      self.r = r
    if k is not None:
      self.k = k
    if length_penalty is not None:
      self.length_penalty = length_penalty
    if use_wt is not None:
      self.use_wt = use_wt
    if use_dis is not None:
      self.use_dis = use_dis
    if wmd_k is not None:
      self.wmd_k = wmd_k
    if wmd_default is not None:
      self.wmd_default = wmd_default
    if use_wmd_estimate is not None:
      self.use_wmd_estimate = use_wmd_estimate
    if use_wmd_memory is not None:
      self.use_wmd_memory = use_wmd_memory
    if debug is not None:
      self.debug = debug
    if prevent_duplicate is not None:
      self.prevent_duplicate = prevent_duplicate

  # Sets the list of stopwords, either from new_list (if it's not None)
  # or from a default list if make_default is True
  # Has no effect if both new_list is None and make_default is False
  def set_stopwords(self, new_list=None, make_default=True):
    if new_list is not None:
        self.stopwords = new_list
    elif make_default:
        self.stopwords = utils.make_stopwords_list()


  """ Functions for reading and initializing data """
  # Read, tokenize, and embed sentences, initializing a lot of data structures
  def embed_sentences(self):
    sens = process_sentences.get_tokenized_sentences(self.data_name)
    self.sen_2_emb = {}
    self.sen_2_ind = {}
    self.ind_2_sen = {}
    self.ind_2_tok_sen = {} 
    for sen in sens:
      res = process_sentences.get_sentence_embedding(sen, self.embedder, self.stopwords, return_tokenized=True)
      if res is not None:
        sen_str = " ".join(sen)
        self.sen_2_emb[sen_str] = res[0]
        ind = len(self.sen_2_emb) - 1
        self.sen_2_ind[sen_str] = ind
        self.ind_2_sen[ind] = sen_str
        self.ind_2_tok_sen[ind] = res[1]

    self.tfidf = TfidfVectorizer(analyzer=lambda x:[w for w in x if w not in self.stopwords])
    self.tfidf.fit([x for x in self.ind_2_tok_sen.values()])

  # Build an index based on the embedded sentences
  def build_sentence_index(self, use_gpu=False):
    if self.sen_2_emb is None:
      self.embed_sentences()
    self.sen_array = np.asarray(list(self.sen_2_emb.values())).astype('float32')
    self.sen_index = faiss.IndexFlatL2(self.sen_array.shape[1])

    if use_gpu:
      gpu = faiss.StandardGpuResources()
      self.sen_index = faiss.index_cpu_to_gpu(gpu, 0, self.sen_index)

    self.sen_index.add(self.sen_array)

  # Embed the words and initialize other data structures
  def embed_words(self):
    self.word_2_emb, self.word_2_pos, self.word_2_count, _ = process_words.get_words(self.data_name, self.embedder, keep_misses=False)
    self.word_2_ind = {k:v for v, k in enumerate(self.word_2_emb.keys())}
    self.ind_2_word = {k:v for v, k in self.word_2_ind.items()}

  # Build an index on the embedded words
  def build_word_index(self, use_gpu=False):
    if self.word_2_emb is None:
      self.embed_words()
    self.word_array = np.asarray(list(self.word_2_emb.values())).astype('float32')
    self.word_index = faiss.IndexFlatL2(self.word_array.shape[1])

    if use_gpu:
      gpu = faiss.StandardGpuResources()
      self.word_index = faiss.index_cpu_to_gpu(gpu, 0, self.word_index)

    self.word_index.add(self.word_array)

  # Given a query string, get word counts and embeddings
  def embed_query(self, query_string):
    text = word_tokenize(query_string)
    word_2_count = dict(Counter(text).most_common())
    word_2_embedding, _, _, _ = process_words.get_dicts(word_2_count, self.embedder, self.stopwords)
    return word_2_embedding, word_2_count

  # Given a query and a list of sentences, remove the query or something extremely close to it from the list
  def remove_duplicate(self, query, sentence_list, edits_allowed=9):
    if query in sentence_list:
      sentence_list.remove(query)
      if self.debug:
        print("Removed exact")
      return sentence_list
    for sen in sentence_list:
      if Levenshtein.distance(query, sen) < edits_allowed:
        if self.debug:
          print("Removed", sen)
        sentence_list.remove(sen)
    return sentence_list
    

  """ Functions for calculating the set cover nearest neighbors """ 
  # Expand the words in the query sentence to include neighbors in the embedding space
  # r is the number of nearest words to include
  def expand_query_sentence(self, query_dict, query_counts):
    # Weight is TF-IDF frequency times number of appearances
    word_wt = query_counts.copy()
    word_wt = {k: v*(1 if k not in self.tfidf.vocabulary_ else self.tfidf.vocabulary_[k]) for k, v in word_wt.items()}

    # Distance from original is 0 for all original words
    word_dis = {k: 0 for k in query_dict.keys()}

    # add plus one to handle cases where words are already in the index
    distances, neighbors = self.word_index.search(np.asarray(list(query_dict.values())), self.r + 1)

    orig_words = list(query_dict.keys())
    for nbrs, dis, key in zip(neighbors, distances, orig_words):
      new_words_count = 0
      for i, d in zip(nbrs, dis):
        # Don't include the r + 1 term if the original term wasn't in the index
        if new_words_count == self.r:
          break
    
        can_word = self.ind_2_word[i]
        # If this isn't a repeat of our target word, add it 
        if can_word != key and can_word not in self.stopwords:
          new_words_count += 1
          # Log the distance
          word_dis[can_word] = 1/(1 + d) if can_word not in word_dis else min(d, word_dis[can_word])

          # Log the weight
          # Related word gets same weight as target word. This is to prevent uncommon synonyms from dominating.
          word_wt[can_word] = word_wt[key]
          
          # Update frequencies
          query_counts[can_word] = 1 if can_word not in query_counts else query_counts[can_word] + 1

    return query_counts, word_wt, word_dis

  # Calculate the score for a given sentence in the database
  def set_cover_score(self, query, db_sentence, word_wt, word_dis):
    score = 0
    matched_words = {}
    for word in set(db_sentence):
      if word in query:
        wt_score = word_wt[word] if self.use_wt else 1
        dis_score = word_dis[word] if self.use_dis else 1
        score += dis_score * wt_score
        matched_words[word] = 1 if word not in matched_words else matched_words[word] + 1
    return score/float(len(db_sentence) ** self.length_penalty), matched_words

  # Run the set-cover algorithm for query sentence
  def get_k_set_cover(self, query):
    # Initialize data if necessary
    if self.word_2_emb is None:
      self.embed_words()
    if self.word_index is None:
      self.build_word_index()
    if self.sen_2_emb is None:
      self.embed_sentences()

    query_emb, query_counts = self.embed_query(query)
    
    # First, get the synonymous words
    query_counts, word_wt, word_dis = self.expand_query_sentence(query_emb, query_counts)
    if self.debug:
      print("Expanded Query:", query_counts)

    candidates = list(self.sen_2_emb.keys())
    if self.prevent_duplicate:
      candidates = self.remove_duplicate(query, candidates)
    found = []
    for _ in range(0, self.k):
      # Track the highest score
      max_score = -1
      max_sen = ""
      max_matches = None

      # Ignore words that have already been fully "covered"
      target_words = [x for x, y in query_counts.items() if y != 0]

      # Iterate over each sentence
      for candidate in candidates:
        score, matches = self.set_cover_score(target_words, candidate.split(" "), word_wt, word_dis)
        if score > max_score:
          max_score = score
          max_sen = candidate
          max_matches = matches
      if self.debug:
        print("Max Score", max_score)
        print("Max Matches", max_matches)
      found.append(max_sen)

      # Update counts of uncovered words & remove the returned sentence from the next calculation
      for word in max_matches.keys():
        query_counts[word] = max(query_counts[word] - max_matches[word], 0)
      candidates.remove(max_sen)

    return found


  """ Functions for calculating other similar sentence techniques """
  # Get the k nearest sentences using the average embedding approach
  def get_k_avg_embed(self, query):
    q_avg = process_sentences.get_sentence_embedding(word_tokenize(query), self.embedder, stopwords=self.stopwords)
    if self.sen_2_emb is None:
      self.embed_sentences()

    if self.sen_index is None:
      self.build_sentence_index()

    nbrs = self.sen_index.search(np.asarray([q_avg]), self.k)[1][0]
    return [self.ind_2_sen[i] for i in nbrs if i != -1]

  # Get the k nearest sentences using any non-index based approach
  def get_k_non_index(self, query, comparison):
        if comparison in ['wmd', "word_movers_distance"]:
          _, q = self.embed_query(query)
          # Map dict to list
          q_list = [[k]*v for k, v in q.items()]
          # Flatten list
          q_list = [v for sublist in q_list for v in sublist]
          # Get representation of all sentences
          candidates = list(self.ind_2_tok_sen.values())
          # We can't index these results, so we have to check exhaustively
          scores = []
          for x in candidates:
            distance_matrix = self.get_wmd_distance_matrix(q_list, x) if self.use_wmd_estimate else None
            scores.append(utils.wmdistance(q_list, x, self.embedder, distance_matrix=distance_matrix))
          # Sort and return top k results
          return [self.ind_2_sen[x] for _, x in sorted(zip(scores, self.ind_2_tok_sen.keys()), key=lambda t: t[0])[:self.k]]

        if comparison == 'jaccard':
          _, q = self.embed_query(query)
          A = set(q.keys())
          # Get representation of all sentences
          candidates = list(self.ind_2_tok_sen.values())
          # We can't index these results, so we have to check exhaustively
          scores = []
          for B in candidates:
            scores.append(utils.jaccard_similarity(A, B))
          # Sort and return top k results
          results = [self.ind_2_sen[x] for _, x in sorted(zip(scores, self.ind_2_tok_sen.keys()), key=lambda t: t[0], reverse=True)[:self.k]]
          if self.debug:
            for r in results:
              _ = utils.jaccard_similarity(A, list(self.ind_2_tok_sen[self.sen_2_ind[r]]), debug=True)
          return results

        if comparison == 'edit_distance':
          # Get representation of all sentences
          candidates = list(self.ind_2_sen.values())
          # We can't index these results, so we have to check exhaustively
          scores = []
          for s in candidates:
            scores.append(Levenshtein.distance(query, s))
          # Sort and return top k results
          return [self.ind_2_sen[x] for _, x in sorted(zip(scores, self.ind_2_sen.keys()), key=lambda t: t[0])[:self.k]]

  """Helper function for estimation of WMD"""
  def get_wmd_distance_matrix(self, d1, d2):
    # Initialize data if necessary
    if self.word_2_emb is None:
      self.embed_words()
    if self.word_index is None:
      self.build_word_index()

    dictionary = Dictionary(documents=[d1, d2])
    vocab_len = len(dictionary)

    # Sets for faster look-up.
    docset1 = set(d1)
    docset2 = set(d2)
    distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
    
    # Use the approximation method
    if self.use_wmd_estimate:
      # If we've just run d1, used cached results
      if self.wmd_prev_query == docset1 and self.wmd_neighbors is not None:
        # Use the cached results
        inds, distances = self.wmd_neighbors
      # If d1 is a new query, get neighbors
      else:
        # Get wmd_k nearest neighbors for each word
        ind_2_emb = {k: np.asarray(self.embedder.emb(v)) for k, v in dictionary.items()}
        inds, distances = self.word_index.search(np.asarray(list(ind_2_emb.values())), self.wmd_k)
        self.wmd_neighbors = (inds, distances)
        self.wmd_prev_query = docset1

      # Populate the distance matrix
      for i, t1 in dictionary.items():
        d = distances[i]
        ind = inds[i]
        for j, t2 in dictionary.items():
          if not t1 in docset1 or not t2 in docset2:
            continue
          # If t2 is a wmd_k nearest neighbor of t1, add its distance
          try:
            j_index = self.word_2_ind[t2]
            distance_matrix[i, j] = d[ind.index(j_index)]
          # Otherwise, make it the default
          except (IndexError, ValueError):
            distance_matrix[i, j] = self.wmd_default

    # Use the dynamic programming approach
    elif self.use_wmd_memory:
      for i, t1 in dictionary.items():
          for j, t2 in dictionary.items():
              if not t1 in docset1 or not t2 in docset2:
                  continue
              # Check if this distance has been calculated
              if (t1, t2) in self.wmd_memory:
                distance_matrix[i, j] = self.wmd_memory[(t1, t2)]
              elif (t2, t1) in self.wmd_memory:
                distance_matrix[i, j] = self.wmd_memory[(t2, t1)]
              # Compute Euclidean distance between word vectors and save it.
              else:
                distance_matrix[i, j] = sqrt(np.sum((np.asarray(self.embedder.emb(t1)) - np.asarray(self.embedder.emb(t2)))**2))
                self.wmd_memory[(t1, t2)] = distance_matrix[i, j]

    # Use original formulation
    else:
      for i, t1 in dictionary.items():
        for j, t2 in dictionary.items():
            if not t1 in docset1 or not t2 in docset2:
                continue
            # Compute Euclidean distance between word vectors and save it.
            else:
              distance_matrix[i, j] = sqrt(np.sum((np.asarray(self.embedder.emb(t1)) - np.asarray(self.embedder.emb(t2)))**2))

    return distance_matrix