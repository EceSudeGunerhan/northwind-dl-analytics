#tahmin fonksiyonu var. modeli yükler ve tamin yapar.

# Bu dosya, eğitilmiş model ve scaler ile yeni müşteri verisi üzerinden tahmin yapar.
# Girdi: total_orders, total_spent, avg_order_value
# Çıktı: Önümüzdeki 6 ayda sipariş verir mi? (0 veya 1)

# Bu dosya model ve scaler ile birlikte tahmin işlemi yapar.

import numpy as np
import joblib
import tensorflow as tf

def load_model_and_scaler():
    model = tf.keras.models.load_model("model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

def predict_customer(model, scaler, total_orders, total_spent, avg_order_value, order_month, order_season):
    X = np.array([[total_orders, total_spent, avg_order_value, order_month, order_season]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0][0]
    return int(prediction >= 0.5)

if __name__ == "__main__":
    model, scaler = load_model_and_scaler()
    result = predict_customer(model, scaler, 4, 950.0, 237.5, 7, 3)
    print("Tahmin sonucu:", "Sipariş verecek" if result == 1 else "Sipariş vermeyecek")
