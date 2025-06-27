import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io
from PIL import Image
import numpy as np

from api.main import app

client = TestClient(app)


# Тесты для корневого эндпоинта
def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "name" in response.json()
    assert "endpoints" in response.json()


# Тесты для эндпоинта классификации
@patch('api.main.load_model')
@patch('api.main.preprocess_image')
@patch('api.main.format_predictions')
def test_classify_endpoint(mock_format, mock_preprocess, mock_load_model):
    mock_preprocess.return_value = np.zeros((1, 224, 224, 3))

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.1, 0.2, 0.7]])
    mock_load_model.return_value = mock_model

    mock_format.return_value = [
        {"class_id": "n01234567", "class_name": "кошка", "probability": 0.7},
        {"class_id": "n01234568", "class_name": "собака", "probability": 0.2},
        {"class_id": "n01234569", "class_name": "птица", "probability": 0.1}
    ]

    test_image = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    response = client.post(
        "/classify/",
        files={"file": ("test_image.jpg", img_byte_arr, "image/jpeg")}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) == 3