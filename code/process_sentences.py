import json
from nltk.corpus import brown, reuters
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import glob
import pandas as pd

def get_tokenized_sentences(dataset):
    if dataset == 'brown_corpus':
        return list(brown.sents())
    elif dataset == 'reuters_corpus':
        return list(reuters.sents())
    elif dataset == 'gatsby':
        with open('./data/gatsby.txt', 'r') as f:
          text = '\n'.join(f.readlines())
        tok_sent = [word_tokenize(t) for t in sent_tokenize(text)]
        return tok_sent
    elif dataset == 'RACE_corpus':
        df_1 = pd.read_csv('./data/middle_combined.csv')
        df_2 = pd.read_csv('./data/high_combined.csv')
        text = '\n'.join(list(df_1['text']) + list(df_2['text']))
        # This dataset appears to have an issue with period spacing
        text = text.replace(".", ". ")
        tok_sent = [word_tokenize(t) for t in sent_tokenize(text)]
        return tok_sent
    elif dataset == 'news':
        df = pd.read_csv('./data/news.csv')
        text = '\n'.join(list(df['content']))
        text = text.replace("   ", " ")
        text = text.replace("   ", " ")
        tok_sent = [word_tokenize(t) for t in sent_tokenize(text)]
        return tok_sent
    elif dataset.startswith("books"):
        difficulty = dataset.split("_")[1]
        files = glob.glob("./data/books/%s/*.txt" % difficulty)
        text = ""
        for file in files:
            with open(file, 'r') as f:
                text += "\n".join(f.readlines()) + "\n"
        # Underscores are used to indicate italics here and should be dropped.        
        text = text.replace("_", "")
        tok_sent = [word_tokenize(t) for t in sent_tokenize(text)]
        return tok_sent
        

# Check to see if word is in the embedding dictionary. If it is, add it to/update dictionaries and return True
# Otherwise return False
def check_and_add(word, embedder, recovered, recovered_str, stopwords):
  # Ignore empty string/stopwords
  if word == '' or word in stopwords:
    return True
  # Otherwise, get its embedding
  embedding = np.asarray(embedder.emb(word)).astype('float32')
  # If we can't embed, return False
  if np.any(np.isnan(embedding)):
    return False
  # Otherwise, log recovered and return True
  else:
    recovered.append(embedding)
    recovered_str.append(word)
    return True

def recover_misses(targets, embedder, stopwords):
    recovered = []
    recovered_str = []
    misses = []
    pre_post = ['_', '.', "'", "'", "*"]
    delims = ['-', '_', '.', '/', ':', "'"]

    while len(targets) > 0:
        word = targets.pop()
        if check_and_add(word, embedder, recovered, recovered_str, stopwords):
            continue

        added = False

        # Check for leading characters
        for c in pre_post:
            if word[0] == c:
                word = word[1:]
                targets.append(word)
                added = True
                break
        if added:
            continue

        # Check for trailing characters
        for c in pre_post:
            if word[-1] == c:
                word = word[:-1]
                targets.append(word)
                added = True
                break
        if added:
            continue

        # Check for words split, for instance by a slash or hyphen
        for c in delims:
            if c in word:
                split_words = word.split(c)
                for s in split_words:
                    targets.append(s)
                added = True
                break
        if added:
            continue

        # Check for capitalization
        if word.isalpha() and not word.islower():
            word = word.lower()
            targets.append(word)
            continue

        # Check for cases where the letter 'l' was misread as the number '1'
        if '1' in word:
            word = word.replace('1', 'l')
            targets.append(word)
            continue
        
        # If none of these have worked, add the original word to misses
        misses.append(word)

    return recovered, misses, recovered_str
        

def get_sentence_embedding(sentence, embedder, stopwords=[], return_tokenized=False):
    embeds = []
    misses = []
    seen = []
    for word in sentence:
        if word not in stopwords:
            emb = np.asarray(embedder.emb(word)).astype('float32')
            if np.any(np.isnan(emb)):
                misses.append(word)
            else:
                embeds.append(emb)
                seen.append(word)

    # Try to recover any misses
    recovered, _, recovered_str = recover_misses(misses, embedder, stopwords)
    for word in recovered_str:
        seen.append(word)
    if len(embeds + recovered) == 0:
        with open("missed_sentences.txt", "a") as f:
            f.write(" ".join(sentence) + '\n')
        return None
    avg = np.mean(embeds + recovered, axis=0)
    if return_tokenized:
        return avg, seen
    return avg

def get_sentence_mappings(sentences, embedder, stopwords=[]):
    sen_2_emb = {}
    for sen in sentences:
        emb = get_sentence_embedding(sen, embedder, stopwords)
        if emb is not None:
            sen_2_emb[' '.join(sen)] = emb
    return sen_2_emb