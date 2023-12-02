from flask import Flask, request, jsonify
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load machine learning models
def load_model():
    # Load SVM model
    svm_model = SVC()
    # Load Logistic Regression model
    lr_model = LogisticRegression()
    # Load Decision Tree model
    dt_model = DecisionTreeClassifier()

    # You may want to load pre-trained weights for the models here if applicable

    return svm_model, lr_model, dt_model

# Route for loading models
@app.route("/load_models", methods=["GET"])
def load_models_route():
    svm_model, lr_model, dt_model = load_model()
    return jsonify({"status": "Models loaded successfully"})

# Your existing routes
@app.route("/", methods=["GET"])
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/", methods=["POST"])
def hello_world_post():
    if not request.is_json:
        return jsonify({"error": "Invalid JSON"}), 400

    suffix = request.json.get("suffix")

    if not suffix:
        return jsonify({"error": "Missing 'suffix' in JSON"}), 400

    return jsonify({"op": f"Hello, World POST {suffix}"})

if __name__ == "__main__":
    app.run(debug=True)
