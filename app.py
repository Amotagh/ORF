


from flask import Flask, render_template,request

from joblib import load
from src.preprocess import clean_text


app = Flask(__name__)

model_tfidf=load(open('./models/model_tfidf.joblib', 'rb'))
model_logistic=load(open('./models/model.joblib', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    text = clean_text(text)
    data = model_tfidf.transform([text])
    pred = model_logistic.predict(data)
    print(text)
    print(pred)

    funding_range = "$5,000 to $150,000" if pred[0] == 0 else "$150,000 to $ 14,000,000"
    return render_template(
        "index.html",
        prediction_text=f'Your project titled "{text}" will revceive between {funding_range} in funding.',
    )


        

    

    