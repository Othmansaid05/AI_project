import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc
)
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# 1. Load data
df = pd.read_csv('data/diabetes.csv')
print("Original data:\n", df.head())

# 2. Handle missing values
medical_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[medical_features] = df[medical_features].replace(0, float('nan'))
imputer = SimpleImputer(strategy='median')
df[medical_features] = imputer.fit_transform(df[medical_features])
print("\nAfter imputation:\n", df.head())

# 3. Outlier detection
print("\nOutlier Detection (IQR Method):")
for feature in medical_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)][feature]
    print(f"{feature} outliers: {len(outliers)}")
    # Optional: Cap outliers (uncomment to apply)
    # df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)

# 4. Visualize feature distributions
plt.figure(figsize=(12, 8))
for i, feature in enumerate(medical_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[feature], kde=True)
    plt.title(f"{feature} Distribution")
plt.tight_layout()
plt.savefig('visualizations/feature_distributions.png')
plt.show()

# 5. Normalize features
X = df.drop('Outcome', axis=1)
y = df['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for deployment
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)

# 7. Hyperparameter tuning
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
print("\nBest parameters:", grid_search.best_params_)
model = grid_search.best_estimator_

# Save model for deployment
pickle.dump(model, open('models/diabetes_model.pkl', 'wb'))

# 8. Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='f1')
print(f"\nCross-Validation F1-Scores: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 9. Predictions and evaluation
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]  # For ROC curve

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Recall:", recall)
print("F1-Score:", f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix Analysis:")
print(f"True Negatives: {tn}, False Positives: {fp}")
print(f"False Negatives: {fn} (missed diabetic cases - critical)")
print(f"True Positives: {tp}")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig('visualizations/confusion_matrix.png')
plt.show()

# 11. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.savefig('visualizations/roc_curve.png')
plt.show()

# 12. Feature importance
feature_names = X.columns
coefficients = model.coef_[0]
sorted_idx = np.argsort(np.abs(coefficients))[::-1]
print("\nFeature Importance (Sorted):")
for idx in sorted_idx:
    print(f"{feature_names[idx]}: {coefficients[idx]:.3f}")
plt.figure(figsize=(8, 6))
sns.barplot(x=coefficients, y=feature_names)
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.savefig('visualizations/feature_importance.png')
plt.show()

# 13. Ethical note
print("\nEthical Note: This model is trained on Pima Indian women and may not generalize to other populations. Validation on diverse datasets is recommended.")