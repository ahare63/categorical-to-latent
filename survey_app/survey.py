from datetime import datetime
import json
import random

# Initialize the questions
def generate_variable_questions(questions, db, var_string):
    question_bank = []
    models = ["set_cover", "weighted_set_cover"]
    baselines = ["embedding_average", "word_movers_distance", "jaccard", "edit_distance"]

    for ind in db.keys():
        # Make sure one of our models is included
        for m in models:
            # Compare it to each baseline
            for b in baselines:
                q_data = {}
                q_data["variable"] = var_string   
                q_data["ind"] = ind
                # Randomly order results
                if random.random() < 0.5:
                    q_data["A_model"] = m
                    q_data["B_model"] = b
                else:
                    q_data["B_model"] = m
                    q_data["A_model"] = b
                question_bank.append(q_data)

    random.shuffle(question_bank)
    return question_bank

# Simple demographic questions that don't rely on the database
def ask_demo_question(question, data, prompt=False):
    print(question["question_text"])
    print(question["options_text"])
    if prompt:
        print(question["options_prompt"])

    valid_response = False
    while not valid_response:
        response = input(">").strip()

        if response in question["valid_inputs"]:
            data[question["title"]] = response
            valid_response = True
            print()
        elif response in question["error_handlers"]:
            data[question["title"]] = question["error_handlers"][response]
            valid_response = True
            print()
        if not valid_response:
            print("Invalid response. Please try again.")
            print(question["question_text"])
            print(question["options_text"])
            print(question["options_prompt"])
    
# More complex questions that revolve around a query
def ask_query_question(db_data, questions, db, data):
    response = db_data.copy()
    db_entry = db[db_data["ind"]]


    # Basic info
    print("Original sentence: ", db_entry["query"])
    dif = db_entry["difficulty"] if db_entry["difficulty"] != "middle" else "pre-high"
    formatted_dif = dif if dif == "college" else dif + " school"
    print("Difficulty level: ", formatted_dif, "\n")

    # Do variable part
    print("Group A sentences:")
    for s in db_entry[db_data["A_model"]].keys():
        print("%d."%(int(s) + 1), db_entry[db_data["A_model"]][s])
    print()
    print("Group B sentences:")
    for s in db_entry[db_data["B_model"]].keys():
        print("%d."%(int(s) + 1), db_entry[db_data["B_model"]][s])
    print()

    q_dict = [x for x in questions["variable"] if x["title"] == db_data["variable"]][0]

    print(q_dict["question_text"])

    valid_response = False
    while not valid_response:
        r = input(">").strip()

        if r in q_dict["valid_inputs"]:
            response["result"] = r
            valid_response = True
            print()
        elif r in q_dict["error_handlers"]:
            response["result"] = q_dict["error_handlers"][r]
            valid_response = True
            print()
        if not valid_response:
            print("Invalid response. Please try again.")
            print(q_dict["question_text"])
            print(q_dict["options_prompt"])

    data["query_responses"].append(response)


# Initialization
data = {}
user_timestamp = str(datetime.now())
data["user_timestamp"] = user_timestamp
data["query_responses"] = []

with open("questions.json", "r") as f:
    questions = json.load(f)
with open("database.json", "r") as f:
    db = json.load(f)

# Randomly pick one variable question
if random.random() < 0.5:
    var_string = "preference"
else:
    var_string = "variety"

question_bank = generate_variable_questions(questions, db, var_string)

# Opening message
print("Welcome! Thank you for agreeing to participate in this study.\n")
print("Before we get started, please answer a few background questions.\n")

for q in questions["opening"]:
    ask_demo_question(q, data, prompt=True)

# Start streaming
print()
print("Now let's begin the study!")
print()
print()
print("Imagine that you’ve been asked to work on a creative writing project and have been provided with a tool that will suggest five sentences related to the one you just wrote. The sentences will be drawn from major works of literature and poetry, grouped into three reading levels (“pre-high school”, “high school”, and “college”). They’re meant to be used for inspiration to help you continue your writing.")
print("Press Enter to continue.")
_ = input(">").strip()
print()
print("The following questions will show you:")
print("1) the original sentence written as part of the project")
print("2) the reading/writing level of the suggested sentences")
print("3) two groups of five sentences meant to help with writing")
print("Press Enter to continue.")
_ = input(">").strip()
print()
print("Please consider the original sentence and suggestions provided and then answer the questions about them. Once you finish the set of questions about one original sentence and its suggestions, either press Enter to load another question or type `quit` and press Enter to be taken to the end of the survey.")
print("Press Enter to continue.")
_ = input(">").strip()
print()

got_quit = False
n_answered = 0
while not got_quit and len(question_bank) > 0:
    q = question_bank.pop()

    print("Question %d\n"%(n_answered+1))
    ask_query_question(q, questions, db, data)
    n_answered += 1

    print("Press Enter to continue or type `quit` and press Enter to jump to the end of the survery.")
    r = input(">").strip()
    if r in ["quit", "Quit", "QUIT"]:
        got_quit = True

# Closing questions
print("You judged %d results!" % n_answered)
for q in questions["closing"]:
    ask_demo_question(q, data)

# Dump the results to json
with open("./responses.json", 'w') as f:
    json.dump(data, f, indent=2)

# Closing message
print("Thank you for your time! Please email the file `responses.json` that has been created in this directory to your instructor.")