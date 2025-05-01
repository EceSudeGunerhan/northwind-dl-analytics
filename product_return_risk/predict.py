# Eğitilmiş modelle tek sipariş için tahmin

# Eğitilmiş model ve scaler ile yeni sipariş verisi üzerinden iade riski tahmini yapar.

import numpy as np
import tensorflow as tf

# Model yükleniyor (scaler artık yok!)
model = tf.keras.models.load_model("model.h5")

# Tahmin fonksiyonu (ölçekleme yok)
def predict_return_risk(model, discount, quantity, total):
    X = np.array([[discount, quantity, total]])
    prediction = model.predict(X)[0][0]
    return int(prediction >= 0.5)

# Örnek test
if __name__ == "__main__":
    discount = 0.25
    quantity = 5
    total = 300.0

    result = predict_return_risk(model, discount, quantity, total)
    print("Tahmin sonucu:", "İade riski yüksek" if result == 1 else "İade riski düşük")
