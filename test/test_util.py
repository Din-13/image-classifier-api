import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
from api.utils import preprocess_image, format_predictions

#
def test_preprocess_image():
    test_image = Image.new('RGB', (100, 100), color='red')

    processed = preprocess_image(test_image)

    assert processed.shape == (1, 224, 224, 3), "Размеры предобработанного изображения некорректны"
    assert isinstance(processed, np.ndarray), "Предобработанное изображение должно быть numpy array"


def test_format_predictions():
    with patch('tensorflow.keras.applications.mobilenet_v2.decode_predictions') as mock_decode:
        mock_decode.return_value = [[('n01234567', 'класс1', 0.7),
                                     ('n01234568', 'класс2', 0.2),
                                     ('n01234569', 'класс3', 0.1)]]

        mock_predictions = np.array([[0.1, 0.2, 0.7]])

        results = format_predictions(mock_predictions, top_k=3)

        assert len(results) == 3, "Должно быть возвращено 3 предсказания"
        assert results[0]['class_name'] == 'класс1', "Имя класса некорректно"
        assert results[0]['probability'] == 0.7, "Вероятность некорректна"