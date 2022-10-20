
#import numpy as np 
from joblib import load

test=['Establishment of the Intelligent Visualization Laboratory']
model_tfidf=load('model_tfidf.joblib')
data=model_tfidf.tranform(test)