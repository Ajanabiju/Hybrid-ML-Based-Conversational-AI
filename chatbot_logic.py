import pickle
import random
import datetime
import pandas as pd

# Load ML model + vectorizer
model = pickle.load(open("ml_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load dataset
df = pd.read_csv("dataset.csv")


def get_bot_response(user_input):
    text = user_input.lower().strip()

    # ✅ RULE-BASED GREETING FIX (ONLY for greeting words)
    if text in ["hi", "hai", "hello", "hey", "hii"]:
        return "Hello! How may I assist you?"

    # ✅ RULE-BASED DATE & TIME
    if "time" in text:
        return "The current time is " + datetime.datetime.now().strftime("%I:%M %p")

    if "date" in text:
        return "Today's date is " + datetime.date.today().strftime("%B %d, %Y")

    # 🔥 ML prediction for ALL OTHER QUERIES
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    max_prob = max(probs)
    tag = model.classes_[probs.argmax()]

    # ❗ LOWER THRESHOLD
    if max_prob < 0.25:
        return "I didn’t understand that. Please try again."

    # Generate response
    responses = df[df["tag"] == tag]["response"].tolist()
    if not responses:
        return "I didn’t understand that. Please try again."

    return random.choice(responses)
