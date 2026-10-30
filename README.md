# Heart Disease Prediction Web Application

A machine learning web application to predict the risk of cardiovascular disease based on clinical patient features.

## Overview
This project uses clinical patient parameters (such as age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, and ST depression) to evaluate heart disease probability using trained classification models.

## Project Structure
- `app.py`: Flask web application providing the prediction interface and scoring endpoint.
- `heart_Disease.ipynb` & `heart6.ipynb`: Exploratory data analysis, model training, feature selection, and model evaluation notebooks.
- `heart.csv`: Heart disease dataset.
- `heart_LG.pkl`: Trained Logistic Regression model.
- `heart_scler.pkl`: StandardScaler transformer fitted on patient features.
- `columns.pkl`: Ordered feature column names used for inference.

## Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/raohuzaifa6262-design/Heart_Disease_pred.git
   cd Heart_Disease_pred
   ```
2. Set up virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate

   pip install flask scikit-learn pandas numpy
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Open `http://localhost:5000` in your browser.
