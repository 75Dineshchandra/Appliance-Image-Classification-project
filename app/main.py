from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

model = tf.keras.models.load_model("app/image_classifier_mc.h5")

class_labels = {
    2: 'TV',
    4: 'Geyser',
    1: 'Refrigerator',
    3: 'Washing Machine',
    0: 'AC'
}

IMG_SIZE = 256  # MUST match training

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image = image.resize((IMG_SIZE, IMG_SIZE))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)

        predicted_label = class_labels.get(predicted_class, "Unknown")

        return {
            "prediction": predicted_label,
            "confidence": float(np.max(prediction))
        }

    except Exception as e:
        return {"error": str(e)}
