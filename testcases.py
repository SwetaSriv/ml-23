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

def test_load_models_route(self):
        response = self.app.get('/load_models')
        data = json.loads(response.get_data(as_text=True))

        self.assertEqual(response.status_code, 200)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'Models loaded successfully')


def test_predict_route_svm(self):
        payload = {"suffix": "test_suffix"}
        response = self.app.post('/predict/svm', json=payload)
        data = json.loads(response.get_data(as_text=True))

        self.assertEqual(response.status_code, 200)
        self.assertIn('op', data)
        self.assertIn('Prediction for svm with suffix', data['op'])

    def test_predict_route_lr(self):
        payload = {"suffix": "test_suffix"}
        response = self.app.post('/predict/lr', json=payload)
        data = json.loads(response.get_data(as_text=True))

        self.assertEqual(response.status_code, 200)
        self.assertIn('op', data)
        self.assertIn('Prediction for lr with suffix', data['op'])

    def test_predict_route_tree(self):
        payload = {"suffix": "test_suffix"}
        response = self.app.post('/predict/tree', json=payload)
        data = json.loads(response.get_data(as_text=True))

        self.assertEqual(response.status_code, 200)
        self.assertIn('op', data)
        self.assertIn('Prediction for tree with suffix', data['op'])





        
