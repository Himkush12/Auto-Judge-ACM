import joblib
import json

def normalize_sample_io(sample_io_raw):
    if not sample_io_raw:
        return ""

    # Case 1: already a string
    if isinstance(sample_io_raw, str):
        return sample_io_raw

    # Case 2: list (strings or dicts)
    if isinstance(sample_io_raw, list):
        parts = []
        for item in sample_io_raw:
            # list of strings
            if isinstance(item, str):
                parts.append(item)

            # list of dicts
            elif isinstance(item, dict):
                if "input" in item:
                    parts.append("Input:\n" + item["input"])
                if "output" in item:
                    parts.append("Output:\n" + item["output"])

        return "\n\n".join(parts)

    # Fallback
    return str(sample_io_raw)


# Load models
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/linear_regressor.pkl")

# Load the SAME problem
INDEX=5
with open("problems_data.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i == INDEX:
            problem = json.loads(line)
            break



title = problem["title"]
description = problem["description"]
input_desc = problem["input_description"]
output_desc = problem["output_description"]
sample_io = normalize_sample_io(problem.get("sample_io", ""))


# EXACT same combination as training
full_text = (
    title.strip() + " " +
    description.strip() + " " +
    input_desc.strip() + " " +
    output_desc.strip() + " " +
    sample_io.strip()
)

X = tfidf.transform([full_text])
predicted_score = model.predict(X)[0]

print("Predicted score (Python):", round(predicted_score, 4))
print("True score (dataset):", problem["problem_score"])
