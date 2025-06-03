from flask import Flask, request, render_template
from pickle import load
import os
import pickle


# Define the Flask app and set the template folder path
app = Flask(__name__, template_folder='../templates')

# Load the trained pipeline
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/pipeline.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

class_dict = {
    0: "Extrovert",
    1: "Introvert"
}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        
        MAX_FRIENDS = 15
        
        # Create dictionary of features from form input
        input_data = {
            'Time_spent_Alone': request.form.get("Time_spent_Alone") or None,
            'Stage_fear': request.form.get("Stage_fear") or None,
            'Social_event_attendance': request.form.get("Social_event_attendance") or None,
            'Going_outside': request.form.get("Going_outside") or None,
            'Drained_after_socializing': request.form.get("Drained_after_socializing") or None,
            'Friends_circle_size': min(float(request.form.get("Friends_circle_size")), MAX_FRIENDS) if request.form.get("Friends_circle_size") else None,
            'Post_frequency': request.form.get("Post_frequency") or None,
        }
        
        if float(request.form.get("Friends_circle_size")) > MAX_FRIENDS:
            message = f"Note: Friend count capped at {MAX_FRIENDS} to stay within model’s trained range."

        # Convert to DataFrame
        import pandas as pd
        input_df = pd.DataFrame([input_data])

        # Predict
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        # Map numeric prediction to label
        pred_class = class_dict[prediction]

        # Get probability for each class label
        proba_dict = {class_dict[i]: f"{round(prob * 100, 1)}%" for i, prob in enumerate(proba)}

    else:
        pred_class = None
        proba_dict = None

    return render_template("index.html", prediction=pred_class, probabilities=proba_dict)


if __name__ == "__main__":
    # Use the port provided by Render, or default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Set host to 0.0.0.0 to be accessible externally
    app.run(host="0.0.0.0", port=port, debug=True)