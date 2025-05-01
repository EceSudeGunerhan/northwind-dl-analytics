# FastAPI uygulaması – Ar-Ge özellikli tahmin API'si
# Girdi: total_orders, total_spent, avg_order_value, order_month, order_season
# Çıktı: {"will_order_again": true/false}

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
import tensorflow as tf

# Model ve scaler'ı yükle
try:
    model = tf.keras.models.load_model("model.h5")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    raise RuntimeError(f"Model veya scaler yüklenemedi: {e}")

# Girdi şeması
class CustomerFeatures(BaseModel):
    total_orders: float = Field(..., example=4)
    total_spent: float = Field(..., example=950.0)
    avg_order_value: float = Field(..., example=237.5)
    order_month: int = Field(..., example=7, ge=1, le=12)
    order_season: int = Field(..., example=3, ge=1, le=4)

    class Config:
        schema_extra = {
            "example": {
                "total_orders": 4,
                "total_spent": 950.0,
                "avg_order_value": 237.5,
                "order_month": 7,
                "order_season": 3
            }
        }

# API başlat
app = FastAPI(
    title="Sipariş Tahmin API",
    description="Müşterinin önümüzdeki 6 ayda sipariş verip vermeyeceğini tahmin eder. (Ar-Ge içerir)",
    version="2.0.0"
)

# Tahmin endpoint'i
@app.post("/predict")
def predict(features: CustomerFeatures):
    try:
        X = np.array([[features.total_orders,
                       features.total_spent,
                       features.avg_order_value,
                       features.order_month,
                       features.order_season]])
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0][0]
        return {"will_order_again": bool(prediction >= 0.5)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {e}")
