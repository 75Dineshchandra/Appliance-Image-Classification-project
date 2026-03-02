from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
import os

app = FastAPI()

MODEL_PATH = "models/best_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = os.listdir("data/train")

@app.get("/")
def read_root():
    return {"message": "Appliance Image Classifier API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    return {"prediction": predicted_class}
