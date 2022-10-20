from flask import Flask
import numpy as np 
from joblib import load

app = Flask(__name__)

@app.route("/")
def hello_world():
    test=['Establishment of the Intelligent Visualization Laboratory']
    model_tfidf=load('model_tfidf.joblib')
    data=model_tfidf.tranform(test)
    return data