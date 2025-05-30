from flask import Flask, request, render_template
from pickle import load
import os


# Define the Flask app and set the template folder path
app = Flask(__name__, template_folder='../templates')

# Get the absolute path to the model file
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/decision_tree_classifier_default_42.sav")
model = load(open(model_path, "rb"))

class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle form submission
        val1 = float(request.form["val1"])
        val2 = float(request.form["val2"])
        val3 = float(request.form["val3"])
        val4 = float(request.form["val4"])

        data = [[val1, val2, val3, val4]]
        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]
    else:
        # Handle initial GET request
        pred_class = None

    # Render the template with the prediction result (or None if GET request)
    return render_template("index.html", prediction=pred_class)


if __name__ == "__main__":
    # Use the port provided by Render, or default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Set host to 0.0.0.0 to be accessible externally
    app.run(host="0.0.0.0", port=port, debug=True)