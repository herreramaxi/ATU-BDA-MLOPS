from flask import Flask, request, render_template
from joblib import load
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)
model=load("SpamModel.pkl")
version = "1.0.1"

@app.route('/version')
def welcome():
    return f"Welcome to Spam Prediction App (version {version})"

@app.route("/")
def home():
    return render_template("index.html", version=version)

@app.route('/predict')
def predict_spam():
    text=request.args.get('text')
    prediction=model.predict([text])
   
    return "The predicted value is" + str(prediction)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]    
    prediction=model.predict([text])
    prediction_text = "Not Spam"

    if prediction == 0:
        prediction_text = "Not Spam"
    elif prediction == 1:
        prediction_text = "Spam"
    else:
        prediction_text = "Unknown"   

    formatted_prediction = f"The predicted value is: {prediction_text}"

    return render_template("result.html", prediction=formatted_prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')