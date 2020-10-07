import glob
import json

# Go through all responses and condense results into single output
def get_response_results():
    with open("results_template.json", 'r') as f:
        summary = json.load(f)
    
    # Iterate through each response
    files = glob.glob("./responses/*.json")
    for f in files:
        with open(f, 'r') as resp_file:
            resp = json.load(resp_file)
        # Top-level questions
        summary["num_responses"] += 1
        summary["major"][resp["major"]] += 1
        summary["frequency"][resp["frequency"]] += 1
        summary["adoption"][resp["adoption"]] += 1
        summary["level"][resp["level"]] += 1

        # Responses for each comparison
        for r in resp["query_responses"]:
            summary[r["A_model"]][r["variable"]][str(r["A_score"])] += 1
            summary[r["B_model"]][r["variable"]][str(r["B_score"])] += 1
            if r["preference"] == "A":
                summary[r["A_model"]]["wins"][r["variable"]][r["B_model"]] += 1
                summary[r["B_model"]]["losses"][r["variable"]][r["A_model"]] += 1
            else:
                summary[r["B_model"]]["wins"][r["variable"]][r["A_model"]] += 1
                summary[r["A_model"]]["losses"][r["variable"]][r["B_model"]] += 1

    # Get averages and win percentages
    for key in ["set_cover", "weighted_set_cover", "embedding_average", "word_movers_distance", "jaccard", "edit_distance"]:
        for val in ["utility", "relevance"]:
            res = summary[key][val]
            summary[key][val]["avg"] = round(sum([int(k)*v for k, v in res.items()])/sum(res.values()), 2) if sum(res.values()) != 0 else 0
        for val in ["utility", "relevance"]:
            wins = sum(summary[key]["wins"][val].values())
            ls = sum(summary[key]["losses"][val].values())
            summary[key]["wins"][val]["win_percentage"] = round(wins/(wins + ls), 2)

    # Save results
    with open("./results.json", 'w') as f:
        json.dump(summary, f, indent=2)

get_response_results()