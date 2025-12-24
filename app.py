from flask import Flask, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("random_forest.pkl")

SHEET_URL = "https://docs.google.com/spreadsheets/d/17fwN4b62EELzsJggFLcH5HJxoNxStgjt6sr9bXOk2dw/export?format=csv"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict")
def predict():
    df = pd.read_csv(SHEET_URL)
    
    latest = df.iloc[-1].to_dict()


    preg = float(latest["Pregnancies"])
    glucose = float(latest["Glucose"])
    bp = float(latest["BloodPressure"])
    skin = float(latest["SkinThickness"])
    insulin = float(latest["Insulin"])
    bmi = float(latest["BMI"])
    dpf = float(latest["DiabetesPedigreeFunction"])
    age = float(latest["Age"])

    metabolic_risk = 0.6 * glucose + 0.4 * bmi

    X = pd.DataFrame([{
        "Pregnancies": preg,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
        "Metabolic_risk": metabolic_risk
    }])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    if pred == 1:
        result = f"🔴 High Diabetes Risk "
    else:
        result = f"🟢 Low Diabetes Risk "

    return render_template("index.html", prediction=result,probability=prob,latest=latest)


if __name__ == "__main__":
    app.run(debug=True)


