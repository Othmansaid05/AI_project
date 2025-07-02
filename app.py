import pickle
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)
model = pickle.load(open('models/diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [float(request.form[f]) for f in feature_names]
            scaled_features = scaler.transform([features])
            prediction = model.predict(scaled_features)[0]
            result = 'Diabetic' if prediction == 1 else 'Non-Diabetic'
            return render_template('result.html', prediction=result)
        except ValueError:
            return render_template('input.html', error="Please enter valid numeric values.")
    return render_template('input.html', error=None)

if __name__ == '__main__':
    app.run(debug=True)