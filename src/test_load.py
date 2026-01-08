import streamlit as st
import joblib

st.title("AutoJudge")
st.write("Predicting Competitive Programming Problem Difficulty")

@st.cache_resource
def load_models():
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    model = joblib.load("models/linear_regressor.pkl")
    return tfidf, model

tfidf_vectorizer, regression_model = load_models()

st.success("Models loaded successfully ✅")
