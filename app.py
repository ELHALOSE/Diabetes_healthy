import numpy as np
from flask import Flask, request, jsonify, render_template,Blueprint
import pickle

# Create flask app
flask_app = Flask(__name__ ,template_folder="template")
model = pickle.load(open("Model.pkl", "rb"))
# site = Blueprint('Site', __name__, template_folder='template') # updated

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = " human diabetes  is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)
