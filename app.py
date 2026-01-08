import streamlit as st
import joblib

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("AutoJudge")
st.write(
    "Estimate the difficulty of a competitive programming problem "
    "using only its textual description."
)

# ----------------------------
# Load models
# ----------------------------


@st.cache_resource
def load_models():
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    model = joblib.load("models/linear_regressor.pkl")
    return tfidf, model


tfidf_vectorizer, regression_model = load_models()

# ----------------------------
# Helper functions
# ----------------------------


def safe_text(x):
    return x.strip() if x else ""


def score_to_class(score):
    if score <= 2.6:
        return "Easy"
    elif score >= 5.8:
        return "Hard"
    else:
        return "Medium"


def class_color(diff_class):
    if diff_class == "Easy":
        return "green"
    elif diff_class == "Medium":
        return "orange"
    else:
        return "red"


# ----------------------------
# Input section
# ----------------------------
st.header("Problem Details")

title = st.text_area(
    "Problem Title (optional)",
    height=80,
    placeholder="e.g. Carnival Wheel"
)

description = st.text_area(
    "Problem Description",
    height=300,
    placeholder="Paste the full problem statement here..."
)

input_desc = st.text_area(
    "Input Description",
    height=150,
    placeholder="Describe the input format..."
)

output_desc = st.text_area(
    "Output Description",
    height=150,
    placeholder="Describe the expected output..."
)

st.divider()
st.header("Sample Input / Output")

sample_input = st.text_area(
    "Sample Input (optional)",
    height=150,
    placeholder="Paste sample input here..."
)

sample_output = st.text_area(
    "Sample Output (optional)",
    height=150,
    placeholder="Paste sample output here..."
)

# ----------------------------
# Action buttons
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    predict_clicked = st.button("Predict Difficulty")

with col2:
    clear_clicked = st.button("Clear All")

if clear_clicked:
    st.experimental_rerun()

# ----------------------------
# Prediction logic
# ----------------------------
if predict_clicked:
    if not safe_text(description):
        st.warning("Please enter at least the problem description.")
    else:
        sample_io = ""
        if safe_text(sample_input) or safe_text(sample_output):
            sample_io = (
                "Input:\n" + safe_text(sample_input) +
                "\n\nOutput:\n" + safe_text(sample_output)
            )

        full_text = (
            safe_text(title) + " " +
            safe_text(description) + " " +
            safe_text(input_desc) + " " +
            safe_text(output_desc) + " " +
            safe_text(sample_io)
        )

        X = tfidf_vectorizer.transform([full_text])
        predicted_score = regression_model.predict(X)[0]
        predicted_class = score_to_class(predicted_score)

        # ----------------------------
        # Results
        # ----------------------------
        st.divider()
        st.subheader("Prediction Result")

        color = class_color(predicted_class)

        st.markdown(
            f"### 🏷 Difficulty: "
            f"<span style='color:{color};'>{predicted_class}</span>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"### 🔢 Difficulty Score: **{round(predicted_score, 2)}**"
        )

        st.caption(
            "This estimate is based only on textual features and may differ "
            "from platform-specific difficulty ratings."
        )
