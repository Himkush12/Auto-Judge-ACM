import json

DATA_PATH = "problems_data.jsonl"
INDEX = 5
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i == INDEX:
            problem = json.loads(line)
            break

print("TITLE:\n", problem["title"], "\n")
print("DESCRIPTION:\n", problem["description"], "\n")
print("INPUT DESCRIPTION:\n", problem["input_description"], "\n")
print("OUTPUT DESCRIPTION:\n", problem["output_description"], "\n")
print("SAMPLE IO:\n", problem["sample_io"], "\n")
print("TRUE SCORE:", problem["problem_score"])
print("TRUE CLASS:", problem["problem_class"])