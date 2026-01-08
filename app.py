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
    if score <= 2.7:
        return "Easy"
    elif score >= 5.4:
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
# Input section + Prediction
# ----------------------------
st.header("Problem Details")

title = st.text_area(
    "Problem Title (optional)",
    height=80
)

description = st.text_area(
    "Problem Description",
    height=300
)

input_desc = st.text_area(
    "Input Description",
    height=150
)

output_desc = st.text_area(
    "Output Description",
    height=150
)

st.divider()
st.header("Sample Input / Output (optional)")

sample_input = st.text_area(
    "Sample Input",
    height=150
)

sample_output = st.text_area(
    "Sample Output",
    height=150
)

predict_clicked = st.button("Predict Difficulty")


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
        # predicted_score = max(0.0, predicted_score)
        predicted_class = score_to_class(predicted_score)

        # ----------------------------
        # Results
        # ----------------------------
        st.divider()
        st.subheader("Prediction Result")

        color = class_color(predicted_class)

        st.markdown(
            f"### Difficulty Class: "
            f"<span style='color:{color};'>{predicted_class}</span>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"### Difficulty Score: **{round(predicted_score, 2)}**"
        )

        st.caption(
            "This estimate is based only on textual features and may differ "
            "from platform-specific difficulty ratings."
        )
