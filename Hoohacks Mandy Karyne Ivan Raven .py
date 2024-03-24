
import pandas as pd
from string import punctuation
from collections import Counter
import re
import matplotlib.pyplot as plt

from flask_cors import CORS
from flask_cors import cross_origin
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from openai import OpenAI

app = Flask(__name__)
# CORS(app)
# CORS(app, supports_credentials=True)
# CORS(app, resources={r"/analyze_dream": {"origins": "https://cdpn.io"}})

print("Received a request to analyze_dream")

tokenizer = AutoTokenizer.from_pretrained(
    "j-hartmann/emotion-english-distilroberta-base"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "j-hartmann/emotion-english-distilroberta-base"
)


def analyze_sentiment(text):
    inputs = tokenizer.encode_plus(
        text, return_tensors="pt", max_length=512, truncation=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)
    scores = scores.numpy()
    max_index = scores.argmax(axis=1)[0]
    max_score = scores[0][max_index]
    labels = model.config.id2label
    predicted_label = labels[max_index]
    return {"label": predicted_label, "score": max_score}


@app.route("/analyze_dream", methods=["POST"])
@cross_origin(
    origins="*", methods=["GET", "POST"], supports_credentials=True
)
def analyze_dream():
    print("analyze_dream")
    dream_description = request.json["dreamDescription"]
    sentiment_result = analyze_sentiment(dream_description)

    my_api_key = "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"
    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to "
                "engage in a helpful, detailed, polite conversation with a user."
            ),
        },
        {
            "role": "user",
            "content": (
                "The following is my dream. Please offer interpretations of the dreams, my underlying emotions, subconscious desires, unresolved conflicts, or simply seeking a deeper understanding of myself. Make it more concise. Give my 3 to 5 bullet points.\n My dream:\n"
                + dream_description
            ),
        },
    ]

    client = OpenAI(api_key=my_api_key, base_url="https://api.perplexity.ai")

    response_stream = client.chat.completions.create(
        model="mistral-7b-instruct",
        messages=messages,
        stream=True,
    )

    all_response = []

    for response in response_stream:
        content = response.choices[0].delta.content
        words = content.split()
        all_response.extend(words)


    final_text = ""
    for word in all_response:
        if word.isdigit():
            final_text += "\n" + word
        elif word.endswith("."):
            final_text += word
        else:
            final_text += " " + word

    final_text = final_text.strip()


    return jsonify(
        {
            "emotion_label": sentiment_result["label"],
            #'emotion_score': sentiment_result['score'],
            "analysis": final_text,
        }
    )
    pass


if __name__ == "__main__":
    app.run(debug=True)
