from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

def assign_priority(text):
    text = text.lower()
    if "refund" in text or "not working" in text or "error" in text:
        return "High"
    elif "how" in text or "change" in text:
        return "Medium"
    else:
        return "Low"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    priority = ""

    if request.method == "POST":
        ticket = request.form["ticket"]
        transformed = vectorizer.transform([ticket])
        prediction = model.predict(transformed)[0]
        priority = assign_priority(ticket)

    return render_template("index.html",
                           prediction=prediction,
                           priority=priority)

if __name__ == "__main__":
    app.run(debug=True)