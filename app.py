from flask import Flask, request, jsonify
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load machine learning models
def load_model(model_type):
    if model_type == "svm":
        return SVC()
    elif model_type == "lr":
        return LogisticRegression()
    elif model_type == "tree":
        return DecisionTreeClassifier()
    else:
        raise ValueError(f"Invalid model type: {model_type}")

# Route for loading models
@app.route("/load_models", methods=["GET"])
def load_models_route():
    svm_model, lr_model, dt_model = load_model("svm"), load_model("lr"), load_model("tree")
    return jsonify({"status": "Models loaded successfully"})

# Route for prediction
@app.route("/predict/<string:model_type>", methods=["POST"])
def predict_route(model_type):
    if not request.is_json:
        return jsonify({"error": "Invalid JSON"}), 400

    suffix = request.json.get("suffix")

    if not suffix:
        return jsonify({"error": "Missing 'suffix' in JSON"}), 400

    try:
        model = load_model(model_type)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # You can now use the 'model' to make predictions based on the input

    return jsonify({"op": f"Prediction for {model_type} with suffix '{suffix}'"})

# Your existing routes
@app.route("/", methods=["GET"])
def hello_world():
    return "<p>Hello, World!</p>"

if __name__ == "__main__":
    app.run(debug=True)
