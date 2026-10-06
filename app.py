from flask import Flask, render_template, request
from transformers import pipeline
import csv
import os
from datetime import datetime

app = Flask(__name__)

# Load AI sentiment model
sentiment_analyzer = pipeline("sentiment-analysis")

LOG_FILE = "sentiment_logs.csv"


def save_log(text, sentiment, confidence):
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Date and Time", "Review", "Sentiment", "Confidence"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            text,
            sentiment,
            f"{confidence:.2f}%"
        ])


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        text = request.form["text"]

        if text.strip():
            prediction = sentiment_analyzer(text)[0]

            sentiment = prediction["label"]
            confidence = prediction["score"] * 100

            result = {
                "label": sentiment,
                "score": confidence
            }

            save_log(text, sentiment, confidence)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)