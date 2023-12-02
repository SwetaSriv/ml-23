from app import app
import pytest
from sklearn.datasets import fetch_openml
import numpy as np
import random

def test_get_root():
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert response.get_data() == b"<p>Hello, World!</p>"

def test_post_root():
    suffix = "post suffix"
    response = app.test_client().post("/", json={"suffix":suffix})
    assert response.status_code == 200    
    assert response.get_json()['op'] == "Hello, World POST "+suffix

def test_load_models_route():
    response = app.test_client().get('/load_models')
    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200
    assert 'status' in data
    assert data['status'] == 'Models loaded successfully'

def test_predict_route_svm():
    payload = {"suffix": "test_suffix"}
    response = app.test_client().post('/predict/svm', json=payload)
    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200
    assert 'op' in data
    assert 'Prediction for svm with suffix' in data['op']

def test_predict_route_lr():
    payload = {"suffix": "test_suffix"}
    response = app.test_client().post('/predict/lr', json=payload)
    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200
    assert 'op' in data
    assert 'Prediction for lr with suffix' in data['op']

def test_predict_route_tree():
    payload = {"suffix": "test_suffix"}
    response = app.test_client().post('/predict/tree', json=payload)
    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200
    assert 'op' in data
    assert 'Prediction for tree with suffix' in data['op']

def test_model_type(m22aie204, solver_name):
    model = load_model(f"{m22aie204}lr{solver_name}.joblib")
    assert isinstance(model, LogisticRegression), "Loaded model is not a Logistic Regression model"

def test_solver_name(m22aie204, solver_name):
    model = load_model(f"{m22aie204}lr{solver_name}.joblib")
    model_solver = model.get_params()['solver']
    assert model_solver == solver_name, f"Solver in the model ({model_solver}) does not match the solver in the file name ({solver_name})"

    @pytest.fixture
def m22aie204():
    # Return the setup or value needed for m22aie204

@pytest.fixture
def solver_name():
