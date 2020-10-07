# This file compiles many functions used across notebooks in order to increase consistency
from collections import Counter
import embeddings
import json
import math
import matplotlib.pyplot as plt
from nltk import pos_tag
from nltk.corpus import brown, reuters
from nltk.tokenize import word_tokenize
import glob
import numpy as np
import pandas as pd
import re
import string
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics

# Return a dictionary mapping each unique token to a count
def get_raw_words(path):
  text = ''
  if path == 'brown_corpus':
    sens = brown.sents()
    text = '\n'.join([' '.join(s) for s in sens])
  elif path == 'reuters_corpus':
    sens = reuters.sents()
    text = '\n'.join([' '.join(s) for s in sens])
  elif path == 'RACE_corpus':
    df_1 = pd.read_csv('./data/middle_combined.csv')
    df_2 = pd.read_csv('./data/high_combined.csv')
    text = '\n'.join(list(df_1['text']) + list(df_2['text']))
    # This dataset appears to have an issue with period spacing
    text = text.replace(".", ". ")
  elif path == 'gatsby':
    with open('./data/gatsby.txt', 'r') as f:
      text = '\n'.join(f.readlines())
  elif path == 'authorship':
    with open('./data/authorship_data.json', 'rb') as f:
      j = json.load(f)
      for key in j.keys():
        text += j[key]['text']
  elif path == 'news_corpus':
    df = pd.read_csv('./data/news.csv')
    text = '\n'.join(list(df['content']))
    text = text.replace("   ", " ")
    text = text.replace("   ", " ")
  elif path.startswith("books"):
    # Get *all* books
    if path == 'books':
        text = ""
        dif = ["middle", "high", "college"]
        for d in dif:
            files = glob.glob(f"./data/books/{d}/*.txt")
            for file in files:
                with open(file, 'r') as f:
                    text += "\n".join(f.readlines()) + "\n"
        # Underscores are used to indicate italics here and should be dropped.        
        text = text.replace("_", "")
            
    else:
        difficulty = path.split("_")[1]
        files = glob.glob("./data/books/%s/*.txt" % difficulty)
        text = ""
        for file in files:
          with open(file, 'r') as f:
            text += "\n".join(f.readlines()) + "\n"
         # Underscores are used to indicate italics here and should be dropped.        
        text = text.replace("_", "")
  # clean text
  text = word_tokenize(text)
  word_2_count = dict(Counter(text).most_common())
  return word_2_count

# For each word (out of context), get the part of speech and map it to the reduced set
def get_pos(word, categories={'noun': 'NN', 'verb': 'VB', 'adverb': 'RB', 
                              'adj': 'JJ', 'pronoun': 'PRP'}):
  tag = pos_tag([word])[0][1]
  for name, val in categories.items():
    if val in tag:
      return name
  return 'other'

# Check to see if word is in the embedding dictionary. If it is, add it to/update dictionaries and return True
# Otherwise return False
def check_and_add(word, embedder, word_2_embedding, word_2_pos, word_2_count, stopwords):
  # Ignore empty string, already seen words, and stopwords
  if word == '' or word in word_2_embedding or word in stopwords:
    return True

  # Otherwise, get its embedding
  embedding = np.asarray(embedder.emb(word)).astype('float32')
  # If we can't embed, return False
  if np.any(np.isnan(embedding)):
    return False
  # Otherwise, update relevant dictionaries and return True
  else:
    word_2_embedding[word] = embedding
    word_2_pos[word] = get_pos(word)
    return True

# Try to recover words which cannot be embedded initially by fixing possible sources of error
def recover_misses(targets, embedder, word_2_embedding, word_2_pos, word_2_count, stopwords):
  pre_post = ['_', '.', "'", "'", "*"]
  delims = ['-', '_', '.', '/', ':', "'"]
  misses = []
  
  # This is designed to prevent issues with duplicates in targets without introducing non-determinism. The set
  # will serve only for reference of what's already in targets. This may pay off as targets is large for some datasets.
  unique_targets = set(targets)

  while len(targets) > 0:
    word = targets.pop()
    unique_targets.remove(word)
    if check_and_add(word, embedder, word_2_embedding, word_2_pos, word_2_count, stopwords):
      continue

    org_word = word
    added = False

    # Check for leading characters
    for c in pre_post:
      if word[0] == c:
        word = word[1:]
        if word not in unique_targets:
          targets.append(word)
          unique_targets.add(word)
        word_2_count[word] = word_2_count[org_word] if word not in word_2_count else word_2_count[word] + word_2_count[org_word]
        del word_2_count[org_word]
        added = True
        break
    if added:
      continue

    # Check for trailing characters
    for c in pre_post:
      if word[-1] == c:
        word = word[:-1]
        if word not in unique_targets:
          targets.append(word)
          unique_targets.add(word)
        word_2_count[word] = word_2_count[org_word] if word not in word_2_count else word_2_count[word] + word_2_count[org_word]
        del word_2_count[org_word]
        added = True
        break
    if added:
      continue

    # Check for words split, for instance by a slash or hyphen
    for c in delims:
      if c in word:
        split_words = word.split(c)
        for s in split_words:
          if s not in unique_targets:
            targets.append(s)
            unique_targets.add(s)
          word_2_count[s] = word_2_count[org_word] if s not in word_2_count else word_2_count[s] + word_2_count[org_word]
        del word_2_count[org_word]
        added = True
        break
    if added:
      continue

    # Check for capitalization
    if word.isalpha() and not word.islower():
      word = word.lower()
      if word not in unique_targets:
        targets.append(word)
        unique_targets.add(word)
      word_2_count[word] = word_2_count[org_word] if word not in word_2_count else word_2_count[word] + word_2_count[org_word]
      del word_2_count[org_word]
      continue

    # Check for cases where the letter 'l' was misread as the number '1'
    if '1' in word:
      word = word.replace('1', 'l')
      if word not in unique_targets:
        targets.append(word)
        unique_targets.add(word)
      word_2_count[word] = word_2_count[org_word] if word not in word_2_count else word_2_count[word] + word_2_count[org_word]
      del word_2_count[org_word]
      continue
      
      # If none of these have worked, add the original word to misses
    misses.append(org_word)

  # Remove any duplicates in misses
  unique_misses = []
  for m in misses:
    if m not in unique_misses:
      unique_misses.append(m)
  
  # Remove empty string from word_2_count
  if '' in word_2_count:
    del word_2_count['']
    
  # Remove stopwords
  for w in stopwords:
    if w in word_2_embedding:
      del word_2_embedding[w]
    if w in word_2_pos:
      del word_2_pos[w]
    if w in word_2_count:
      del word_2_count[w]
  return word_2_embedding, word_2_pos, word_2_count, unique_misses

# This function maps each word to an embedding, a part of speech, and a count in the original text
def get_dicts(word_2_count, embedder, stopwords):
  word_2_embedding = {}
  word_2_pos = {}
  misses = []
  for word in word_2_count.keys():
    if word not in stopwords:
        # Log words that could not be embedded
        embedding = np.asarray(embedder.emb(word)).astype('float32')
        if np.any(np.isnan(embedding)):
          misses.append(word)
          continue
        word_2_embedding[word] = embedding
        word_2_pos[word] = get_pos(word)

  return recover_misses(misses, embedder, word_2_embedding, word_2_pos, word_2_count, stopwords)

# Returns words_2_embedding, word_2_pos, misses
def get_words(dataset, embedder, keep_misses=True, stopwords=[]):
    word_2_count = get_raw_words(dataset)
    word_2_embedding, word_2_pos, word_2_count, unique_misses = get_dicts(word_2_count, embedder, stopwords)
    if not keep_misses:
      word_2_count = remove_misses(word_2_count, unique_misses, word_2_embedding)
    return word_2_embedding, word_2_pos, word_2_count, unique_misses

# Remove misses from word_2_count and assert the keys are the same as those in word_2_embedding
def remove_misses(word_2_count, misses, word_2_embedding):
  for word in misses:
    if word in word_2_count:
      del word_2_count[word]
    else:
      print("Missing", word)
  assert(set(word_2_count.keys()) == set(word_2_embedding.keys()))
  return word_2_count

def embed_query(query_string, embedder, stopwords):
  text = word_tokenize(query_string)
  word_2_count = dict(Counter(text).most_common())
  word_2_embedding, _, _, _ = get_dicts(word_2_count, embedder)
  return word_2_embeddings, word_2_count
  
    