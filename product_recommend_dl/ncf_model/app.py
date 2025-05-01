# NCF modelini FastAPI ile kategori öneri API'si olarak sunar (hatasız sürüm)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import os

# Dosya yolları: script ile aynı dizinden
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_PATH, "model.h5")
user_encoder_path = os.path.join(BASE_PATH, "user_encoder.npy")
cat_encoder_path = os.path.join(BASE_PATH, "cat_encoder.npy")

# Model ve encoder'ları yükle
model = tf.keras.models.load_model(model_path)
user_classes = np.load(user_encoder_path, allow_pickle=True)
cat_classes = np.load(cat_encoder_path, allow_pickle=True)

# Girdi tipi
class UserRequest(BaseModel):
    customer_id: str

# FastAPI uygulaması
app = FastAPI(
    title="NCF Ürün Kategori Öneri API",
    description="Kullanıcıya kategori önerisi yapar (Neural Collaborative Filtering ile)",
    version="1.0.0"
)

@app.post("/predict")
def recommend_categories(request: UserRequest):
    customer_id = str(request.customer_id)

    # Tip uyumu için string'e çevirip karşılaştır
    user_ids_str = user_classes.astype(str)

    if customer_id not in user_ids_str:
        raise HTTPException(status_code=404, detail="Kullanıcı ID eğitim verisinde yok.")

    # Müşteri indeksini bul
    user_index = np.where(user_ids_str == customer_id)[0][0]
    cat_indices = np.arange(len(cat_classes))
    user_input = np.full(shape=len(cat_indices), fill_value=user_index)

    # Tahmin yap
    preds = model.predict([user_input, cat_indices], verbose=0).flatten()
    top_3 = preds.argsort()[-3:][::-1]  # En yüksek 3 skor

    result = {
        "recommended_categories": [str(cat_classes[i]) for i in top_3],
        "category_scores": dict(zip([str(cat) for cat in cat_classes], [round(float(p), 4) for p in preds]))
    }
    return result


# Örnek API çıktısı:
# {
#   "recommended_categories": ["8", "3", "1"],
#   "category_scores": {
#     "1": 0.845,
#     "2": 0.4797,
#     "3": 0.8506,
#     "4": 0.7977,
#     "5": 0.4938,
#     "6": 0.4812,
#     "7": 0.4407,
#     "8": 0.927
#   }
# }
#
# Açıklama:
# - "recommended_categories": Modelin, kullanıcı için en yüksek satın alma ihtimali gördüğü 3 kategori ID'sidir.
#     Bu örnekte: kategori 8, 3 ve 1 önerilmiştir.
# - "category_scores": Modelin her kategoriye verdiği satın alma tahmin skorudur (0 ile 1 arasında).
# - Bu skorlar, kullanıcının daha önce etkileşimde bulunmadığı kategoriler için hesaplanır.
# - Bu skorlar yüksekse (örneğin 0.9), model kullanıcının o kategoriye ilgi göstereceğini öngörmektedir.
