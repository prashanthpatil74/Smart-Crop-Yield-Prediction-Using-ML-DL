import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load feature column names (used in training)
X_ohe_columns = pickle.load(open('X_columns.pkl', 'rb'))  # Ensure this file exists

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from the form
        Rainfall_mm = float(request.form['Rainfall_mm'])
        Temperature_Celsius = float(request.form['Temperature_Celsius'])
        Fertilizer_Used = 1 if request.form['Fertilizer_Used'] == 'True' else 0
        Irrigation_Used = 1 if request.form['Irrigation_Used'] == 'True' else 0
        Days_to_Harvest = int(request.form['Days_to_Harvest'])

        # One-hot encode categorical features
        Region = request.form['Region']
        Soil_Type = request.form['Soil_Type']
        Crop = request.form['Crop']
        Weather_Condition = request.form['Weather_Condition']

        # Create a DataFrame
        input_data = pd.DataFrame({
            'Rainfall_mm': [Rainfall_mm],
            'Temperature_Celsius': [Temperature_Celsius],
            'Fertilizer_Used': [Fertilizer_Used],
            'Irrigation_Used': [Irrigation_Used],
            'Days_to_Harvest': [Days_to_Harvest],
            'Region': [Region],
            'Soil_Type': [Soil_Type],
            'Crop': [Crop],
            'Weather_Condition': [Weather_Condition]
        })

        # Apply one-hot encoding
        X_ohe_input = pd.get_dummies(input_data, columns=['Region', 'Soil_Type', 'Crop', 'Weather_Condition'], drop_first=True)

        # Ensure input columns match training columns
        missing_cols = set(X_ohe_columns) - set(X_ohe_input.columns)
        for c in missing_cols:
            X_ohe_input[c] = 0

        # Reorder columns
        X_ohe_input = X_ohe_input[X_ohe_columns]

        # Make prediction
        prediction = model.predict(X_ohe_input)
        prediction_text = f"Predicted Crop Yield: {prediction[0]:,.2f}"

    except Exception as e:
        prediction_text = f"Error: {e}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)
    #app.run(host="127.0.0.1", port=9000)