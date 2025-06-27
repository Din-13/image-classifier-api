from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from .utils import preprocess_image, format_predictions

app = FastAPI(title="Классификатор изображений API")

model = None

@app.on_event("startup")
async def startup():
    load_model()

def load_model():
    global model
    if model is None:
        model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # изменить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/classify/", response_class=JSONResponse)
async def classify_image(file: UploadFile = File(...)):
    allowed_extensions = ["jpg", "jpeg", "png"]
    file_extension = file.filename.split(".")[-1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Файл должен быть одного из следующих типов: {', '.join(allowed_extensions)}"
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        processed_image = preprocess_image(image)

        model = load_model()

        predictions = model.predict(processed_image)

        results = format_predictions(predictions)

        return {
            "status": "success",
            "predictions": results
        }

    except Exception as e:
        print(f"Ошибка при обработке изображения: {str(e)}")  # Для отладки
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке изображения: {str(e)}")


@app.get("/", response_class=JSONResponse)
async def root():
    return {
        "name": "Классификатор изображений API",
        "description": "API для классификации изображений с помощью MobileNetV2",
        "endpoints": {
            "/classify/": "Загрузка и классификация изображений"
        }
    }