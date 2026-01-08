# AutoJudge – Predicting Programming Problem Difficulty

## Project Overview
Competitive programming platforms categorize problems into difficulty levels
(Easy / Medium / Hard) and often assign a numerical difficulty score. These labels
are typically decided manually based on experience and community feedback.

**AutoJudge** is a machine learning–based system that automatically predicts:
- The **difficulty class** (Easy / Medium / Hard)
- A **numerical difficulty score**

The prediction is performed using **only the textual description of the problem**,
without relying on submission statistics or user performance data.

---

## Dataset Used
The project uses a dataset of competitive programming problems stored in JSONL format.

Each problem contains:
- `title`
- `description`
- `input_description`
- `output_description`
- `problem_class` (Easy / Medium / Hard)
- `problem_score` (numerical difficulty value)

The dataset is located at:
data/problems_data.jsonl


---

## Approach

### 1. Data Preprocessing
- Relevant textual fields (`title`, `description`, `input_description`,
  `output_description`) are cleaned and concatenated into a single text input.
- Missing or optional fields are handled safely to avoid runtime errors.

### 2. Feature Extraction
- **TF-IDF (Term Frequency–Inverse Document Frequency)** is used to convert text into
  numerical feature vectors.
- This representation captures the importance of keywords commonly associated with
  problem difficulty (e.g., algorithmic terms, constraints, and technical language).

### 3. Models Used

#### Regression Model
- **Linear Regression**
- Predicts a continuous numerical difficulty score based on TF-IDF features.

#### Difficulty Class Mapping
- The predicted numerical score is converted into a difficulty class
  (**Easy / Medium / Hard**) using empirically chosen thresholds.
- These thresholds were selected by analyzing the distribution of difficulty scores
  in the training dataset, ensuring consistency with the original labels.
- This approach keeps classification logic simple, interpretable, and aligned with
  the regression output.

---

## Other Models Explored
During experimentation, multiple machine learning models were evaluated before
finalizing the approach:

- Logistic Regression (classification)
- Random Forest (classification and regression)
- Support Vector Machine (classification)
- XGBoost (gradient boosting)

Although some of these models achieved competitive results, they did not provide
significant improvement over the final approach when considering both performance
and interpretability.

### Comparative Performance Summary
- TF-IDF + Linear Regression:
  - Provided stable predictions and strong overall performance
  - Chosen for simplicity and interpretability
- Random Forest / XGBoost:
  - Slight improvements in some cases
  - Higher complexity and less interpretability
- Logistic Regression / SVM:
  - Lower classification accuracy compared to the final pipeline

Based on these observations, **TF-IDF with Linear Regression** was selected as the
final model due to its balance of accuracy, simplicity, and explainability.

---

## Saved Trained Models
The trained models used for prediction are saved and included directly in the repository:

- TF-IDF Vectorizer
- Linear Regression model for difficulty score prediction

These models are loaded in the web application without retraining, ensuring fast and
consistent predictions.

---

## Evaluation Metrics

### Classification
- Accuracy
- Confusion Matrix

### Regression
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

All reported metrics correspond to the trained models included in this repository.

---

## Web Interface
A simple **Streamlit** web interface allows users to:
1. Paste a problem description
2. Provide input/output descriptions (optional)
3. Provide sample input/output data (optional)
3. Click **Predict Difficulty**
4. View:
   - Predicted difficulty class
   - Predicted numerical difficulty score

The interface runs entirely locally and does not require deployment or hosting.

---

## How to Run the Project Locally

### 1. Clone the repository
```bash
git clone https://github.com/Himkush12/Auto-Judge-ACM.git
cd Auto-Judge-ACM
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web application
```bash
streamlit run app.py
```

## Demo Video
A short demo video (2–3 minutes) demonstrating:
- Project overview
- Model approach
- Working web interface with sample predictions

 **Demo Video Link:**  
(Add link here)

## Author
**Himesh Kushwaha**  
24410011
Btech 2nd Yr
Mechanical Engineering  

