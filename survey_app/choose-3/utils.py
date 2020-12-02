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
        if resp["survey_type"] == "choose-3":
            # Top-level questions
            summary["num_responses"] += 1
            summary["major"][resp["major"]] += 1
            summary["frequency"][resp["frequency"]] += 1
            summary["adoption"][resp["adoption"]] += 1
            summary["level"][resp["level"]] += 1
            if not include_never and resp["frequency"] == "A":
                continue

            # Responses for each comparison
            for r in resp["query_responses"]:
                if "weighted_set_cover" in r["option_to_model"].values():
                    summary["num_answered_weighted"] += 1
                    for inc in r["result"]:
                        model = r["option_to_model"][inc]
                        summary[model]["included_count_weighted"] += 1
                else:
                    summary["num_answered_unweighted"] += 1
                    for inc in r["result"]:
                        model = r["option_to_model"][inc]
                        summary[model]["included_count_unweighted"] += 1
                

    # Get averages and win percentages
    for key in ["set_cover", "weighted_set_cover", "embedding_average", "word_movers_distance", "jaccard", "edit_distance"]:
        if key != "set_cover":
            summary[key]["included_pct_weighted"] = round(summary[key]["included_count_weighted"]/summary["num_answered_weighted"], 2)
        if key != "weighted_set_cover":
            summary[key]["included_pct_unweighted"] = round(summary[key]["included_count_unweighted"]/summary["num_answered_unweighted"], 2)

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


if __name__ == "__main__":
    get_response_results()