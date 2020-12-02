import glob
import json

# Go through all responses and condense results into single output
def get_response_results(include_never=True):
    with open("results_template.json", 'r') as f:
        summary = json.load(f)
    
    # Iterate through each response
    files = glob.glob("./responses/*.json")
    for f in files:
        with open(f, 'r') as resp_file:
            resp = json.load(resp_file)
        # Top-level questions
        if resp["survey_type"] == "head-to-head":
            summary["num_responses"] += 1
            summary["major"][resp["major"]] += 1
            summary["frequency"][resp["frequency"]] += 1
            summary["adoption"][resp["adoption"]] += 1
            summary["level"][resp["level"]] += 1
            if not include_never and resp["frequency"] == "A":
                continue

            # Responses for each comparison
            for r in resp["query_responses"]:
                if r["result"] == "A":
                    summary[r["A_model"]]["wins"][r["variable"]][r["B_model"]] += 1
                    summary[r["B_model"]]["losses"][r["variable"]][r["A_model"]] += 1
                else:
                    summary[r["B_model"]]["wins"][r["variable"]][r["A_model"]] += 1
                    summary[r["A_model"]]["losses"][r["variable"]][r["B_model"]] += 1

    # Get averages and win percentages
    for key in ["set_cover", "weighted_set_cover", "embedding_average", "word_movers_distance", "jaccard", "edit_distance"]:
        for val in ["variety", "preference"]:
            wins = sum(summary[key]["wins"][val].values())
            ls = sum(summary[key]["losses"][val].values())
            summary[key]["wins"][val]["win_percentage"] = round(wins/(wins + ls), 2) if wins + ls > 0 else 0

    # Save results
    with open("./results.json", 'w') as f:
        json.dump(summary, f, indent=2)


# Take data in new_file, add any additional data in new_file to it, and save as new_file
def update_database(old_file, new_file):
    with open(old_file, 'r') as f:
        old = json.load(f)
    with open(new_file, 'r') as f:
        new = json.load(f)

    for key in new.keys():
        new_dict = new[key]
        old_dict = old[key]

        for k in old_dict.keys():
            if k not in new_dict:
                new_dict[k] = old_dict[k]
    
    with open(new_file, 'w') as f:
        json.dump(new, f, indent=2)

def jaccard_similarity(A, B, debug=False):
    A = set(A)
    B = set(B)
    intersection = len(A.intersection(B))
    if debug:
        print("Intersection", intersection)
    union = len(A) + len(B) - intersection
    return intersection/union

def make_stopwords_list():
  stopwords = list(nltk_stopwords.words('english'))
  # Add capitalized version of stopwords. str.title() messes up contractions
  stopwords += list([x[0].upper() + x[1:] for x in stopwords])
  # Add punctuation
  stopwords += [".", ",", "!", "?", "...", "-", ":", ";", "“"]
  return stopwords

# Get the average Jaccard similarity between each response

# With stop: {'set_cover': 0.06754981781619317, 'weighted_set_cover': 0.07185264873518071, 'embedding_average': 0.10510930115263878, 'word_movers_distance': 0.1436753000213796, 'jaccard': 0.17353139712257995, 'edit_distance': 0.13754881746997755}
# Without stop: {'set_cover': 0.005680999555999555, 'weighted_set_0cover': 0.006416638916638916, 'embedding_average': 0.07044135641620161, 'word_movers_distance': 0.1540660031623887, 'jaccard': 0.26483285603285606, 'edit_distance': 0.027155863988952225}
def avg_jaccard(stopwords=True):
    sims = {}
    sims["set_cover"] = []
    sims["weighted_set_cover"] = []
    sims["embedding_average"] = []
    sims["word_movers_distance"] = []
    sims["jaccard"] = []
    sims["edit_distance"] = []
    with open('./database.json', 'r') as f:
        db = json.load(f)
    if stopwords:
        stopwords = set(make_stopwords_list())
    else:
        stopwords = set()
    for resp in db.keys():
        for model in ["set_cover", "weighted_set_cover", "embedding_average", "word_movers_distance", "jaccard", "edit_distance"]:
            sens = list(db[resp][model].values())
            for i, s_1 in enumerate(sens):
                left = set(word_tokenize(s_1)).difference(stopwords)
                for s_2 in sens[i+1:]:
                    right = set(word_tokenize(s_2)).difference(stopwords)
                    sims[model].append(jaccard_similarity(left, right))
    print({k: np.mean(v) for k, v in sims.items()})

def unique_sentences():
    with open('./database.json', 'r') as f:
        db = json.load(f)
    models = ["set_cover", "embedding_average", "word_movers_distance", "jaccard", "edit_distance"]
    unique = {}
    for m in models:
        unique[m] = 0
    for resp in db.keys():
        for model in models:
            sens = list(db[resp][model].values())
            other_sens = set()
            o = [list(db[resp][m].values()) for m in models if m != model]
            for l in o:
                for s in l:
                    other_sens.add(s)
            for i, s in enumerate(sens):
                unique_for_alg = s not in sens[:i] and s not in sens[i+1:]
                unique_for_query = s not in other_sens
                if unique_for_alg and unique_for_query:
                    unique[model] += 1
    for m in models:
        unique[m] =  unique[m]/(5*len(db.keys()))
    print(unique)

def avg_questions_answered():
    # Iterate through each response
    files = glob.glob("./responses/*.json")
    h2h = []
    c3 = []
    for f in files:
        with open(f, 'r') as resp_file:
            resp = json.load(resp_file)
        if resp["survey_type"] == "head-to-head":
            h2h.append(len(resp["query_responses"]))
        elif resp["survey_type"] == "choose-3":
            c3.append(len(resp["query_responses"]))
    print("h2h", np.mean(h2h))
    print("c3", np.mean(c3))


if __name__ == "__main__":
    # avg_jaccard()
    # get_response_results()
    # unique_sentences()
    avg_questions_answered()