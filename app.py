from flask import Flask, render_template, request
from transformers import pipeline
from flask_pymongo import PyMongo
from config import MONGO_URI
from datetime import datetime

app = Flask(__name__)

# MongoDB Configuration
app.config["MONGO_URI"] = MONGO_URI
mongo = PyMongo(app)

# Load AI Model
sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        text = request.form["text"]

        prediction = sentiment(text)[0]

        label = prediction["label"]
        confidence = round(prediction["score"] * 100, 2)

        result = {
            "text": text,
            "label": label,
            "score": confidence
        }

        # Save to MongoDB
        mongo.db.history.insert_one({
            "text": text,
            "sentiment": label,
            "confidence": confidence,
            "date": datetime.now()
        })

    return render_template("index.html", result=result)


@app.route("/history")
def history():

    data = list(mongo.db.history.find().sort("date", -1))

    return render_template("history.html", history=data)


if __name__ == "__main__":
    app.run(debug=True)