AI-Based Diabetes Prediction Project
Overview
This project develops an interpretable AI model to predict diabetes risk using the Pima Indian Diabetes Dataset. It uses logistic regression, emphasizing clinical interpretability for healthcare applications.
Folder Structure

data/: Contains diabetes.csv (Pima Indian Diabetes Dataset).
src/: Main script (diabetes_project.py) for data preprocessing, model training, and evaluation.
templates/: HTML files for Flask web app.
models/: Saved model and scaler (diabetes_model.pkl, scaler.pkl).
reports/: Deliverables for Phases 1–4.
slides/: Presentation slides for Phase 5.
visualizations/: Saved plots (confusion matrix, ROC curve, feature importance).

Setup

Clone the repository: git clone <repository-link>
Install dependencies: pip install -r requirements.txt
Place diabetes.csv in the data/ folder.
Run the main script: python src/diabetes_project.py
(Optional) Run the Flask app: python app.py

Dataset

Source: UCI Machine Learning Repository via Kaggle
Features: 8 clinical measurements (e.g., Glucose, BMI, Age)
Target: Binary (0 = Non-Diabetic, 1 = Diabetic)
Size: 768 records

Model

Algorithm: Logistic Regression
Justification: Interpretable coefficients, low computational cost, suitable for binary classification.
Preprocessing: Median imputation for missing values, StandardScaler for normalization, outlier detection (IQR).
Evaluation: Accuracy, recall, F1-score, confusion matrix, ROC curve (AUC).

Ethical Note
The model is trained on Pima Indian women and may not generalize to other populations. Validation on diverse datasets is recommended.
Repository

GitHub: https://github.com/Othmansaid05/AI_project
