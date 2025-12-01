from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your trained model
model = joblib.load('model.joblib')

# List of features in order
feature_names = [
    'age', 'height(cm)', 'weight(kg)', 'waist(cm)',
    'eyesight(left)', 'eyesight(right)', 'hearing(left)', 'hearing(right)',
    'systolic', 'relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride',
    'HDL', 'LDL', 'hemoglobin', 'Urine protein', 'serum creatinine',
    'AST', 'ALT', 'Gtp', 'dental caries'
]

# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/')
def home():
    return render_template('index.html', form_values={}, prediction_text=None)


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         input_values = [float(request.form[feature]) for feature in feature_names]
#         input_df = pd.DataFrame([input_values], columns=feature_names)
#         prediction = model.predict(input_df)[0]
#         prediction_text = f"Prediction: {prediction}"
#         if prediction == 1:
#            prediction_text =  "Prediction: 0 => The person is a smoker"
#         else:
#             prediction_text = "Prediction: 1 => The person is not a smoker"
#     except Exception as e:
#         prediction_text = f"Error: {str(e)}"
    
#     return render_template('index.html', prediction_text=prediction_text)

@app.route('/predict', methods=['POST'])
def predict():
    form_values = {}
    try:
        for feature in feature_names:
            value = request.form[feature]
            form_values[feature] = value  # store for redisplay
        input_values = [float(form_values[feature]) for feature in feature_names]
        input_df = pd.DataFrame([input_values], columns=feature_names)
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            prediction_text = "Prediction: The person is a smoker"
        else:
            prediction_text = "Prediction: The person is not a smoker"
    except Exception as e:
        prediction_text = f"Error: {str(e)}"
    
    return render_template('index.html', prediction_text=prediction_text, form_values=form_values)


if __name__ == '__main__':
    app.run(debug=True)
