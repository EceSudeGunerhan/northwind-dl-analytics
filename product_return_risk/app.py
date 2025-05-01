# Derin öğrenme modeli ile iade riski tahmini yapan REST API

from fastapi import FastAPI
from pydantic import BaseModel, Field
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model.h5")

class OrderFeatures(BaseModel):
    discount: float = Field(..., example=0.25, ge=0.0, le=1.0)
    quantity: int = Field(..., example=5, gt=0)
    total: float = Field(..., example=300.0, gt=0)

    class Config:
        schema_extra = {
            "example": {
                "discount": 0.25,
                "quantity": 5,
                "total": 300.0
            }
        }

app = FastAPI(
    title="Ürün İade Riski API",
    description="Bir siparişin iade edilme riskini tahmin eder.",
    version="1.0.0"
)

@app.post("/predict")
def predict(features: OrderFeatures):
    X = np.array([[features.discount, features.quantity, features.total]])
    prediction = model.predict(X)[0][0]
    return {
        "will_return": bool(prediction >= 0.5),
        "risk_score": round(float(prediction), 4)
    }
