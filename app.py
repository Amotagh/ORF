from operator import methodcaller
import re
from flask import Flask, render_template,request
import pandas as pd
import numpy as np
import sklearn
from joblib import load

app = Flask(__name__)

@app.route("/",methods=['GET','POST'])
def hello_world():
    request_type_str=request.method
    if request_type_str == 'GET':
        return render_template('index.html')
    
        text=request.form['text']
        model_tfidf=load('model_tfidf.joblib')
        data=model_tfidf.transform(text)
        model_logistic=load('model.joblib')
        pred = model_logistic.predict (data)
        if pred[0] == 0:
             return render_template('index0.html')  
        else:
             return render_template('index1.html')


        

    

    