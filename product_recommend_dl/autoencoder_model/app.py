from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

# Eğitilmiş modeli yükle
model = tf.keras.models.load_model("model.h5")

CATEGORY_NAMES = [
    "Beverages", "Condiments", "Confections", "Dairy Products",
    "Grains/Cereals", "Meat/Poultry", "Produce", "Seafood"
]

# JSON giriş şeması
class CategorySpending(BaseModel):
    Beverages: float
    Condiments: float
    Confections: float
    Dairy_Products: float
    Grains_Cereals: float
    Meat_Poultry: float
    Produce: float
    Seafood: float

# Uygulama başlatılıyor
app = FastAPI()

@app.post("/predict")
def predict(features: CategorySpending):
    X = np.array([[
        features.Beverages,
        features.Condiments,
        features.Confections,
        features.Dairy_Products,
        features.Grains_Cereals,
        features.Meat_Poultry,
        features.Produce,
        features.Seafood
    ]])
    preds = model.predict(X)[0]
    return {
        "recommended_categories": [cat for cat, p in zip(CATEGORY_NAMES, preds) if p >= 0.5],
        "category_scores": dict(zip(CATEGORY_NAMES, [round(float(p), 4) for p in preds]))
    }


# Örnek API çıktısı açıklaması:
# {
#   "recommended_categories": [
#     "Beverages",
#     "Confections"
#   ],
#   "category_scores": {
#     "Beverages": 0.5934,
#     "Condiments": 0.0092,
#     "Confections": 0.9235,
#     "Dairy Products": 0.0028,
#     "Grains/Cereals": 0.1172,
#     "Meat/Poultry": 0.1492,
#     "Produce": 0.1263,
#     "Seafood": 0.0015
#   }
# }
#
# Açıklama:
# - "category_scores": Modelin her kategori için satın alma olasılığı tahminidir (0 ile 1 arası).
# - "recommended_categories": Tahmin skoru 0.5 veya üzeri olan kategoriler öneri listesine alınmıştır.
# - Bu örnekte model, "Beverages" (%59) ve "Confections" (%92) kategorilerinde yeni ürün alma potansiyelini yüksek bulmuştur.
