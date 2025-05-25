from flask import Flask, request
from joblib import load
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)
model=load("SpamModel.pkl")
version = "1.0.0"


@app.route('/')
def welcome():
    return f"Welcome to Spam Prediction App (version {version})"

@app.route('/predict')
def predict_spam():
    experience=request.args.get('text')
    prediction=model.predict([[experience]])
   
    return "The predicted value is" + str(prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')