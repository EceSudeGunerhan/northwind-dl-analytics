# Bir müşterinin kategori bazlı satın alma potansiyelini tahmin eder ve görselleştirir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from database import get_customer_category_matrix

# Model yükleniyor
model = tf.keras.models.load_model("autoencoder_model/model.h5")

THRESHOLD = 0.5

def predict_for_customer(customer_index=0):
    df = get_customer_category_matrix()
    customer_id = df.loc[customer_index, "customer_id"]
    X = df.drop(columns=["customer_id"])
    category_names = X.columns

    # Girdi vektörü hazırlanıyor
    input_vector = X.iloc[customer_index].values.reshape(1, -1)
    preds = model.predict(input_vector)[0]

    # Önerilen kategoriler (0.5 üzeri)
    recommended_categories = [cat for cat, p in zip(category_names, preds) if p >= THRESHOLD]

    # erminal çıktısı
    print(f"\nMüşteri ID: {customer_id}")
    print("Önerilen Kategoriler:")
    for cat in recommended_categories:
        print(f" - {cat}")
    print(f"Tahmin Skorları: {dict(zip(category_names, np.round(preds, 2)))}")

    # Görselleştirme
    plt.figure(figsize=(10, 5))
    plt.bar(category_names, preds, color=["green" if p >= THRESHOLD else "gray" for p in preds])
    plt.axhline(y=THRESHOLD, color='red', linestyle='--', label=f"Eşik ({THRESHOLD})")
    plt.xticks(rotation=45)
    plt.ylabel("Tahmin Skoru")
    plt.title(f"Müşteri: {customer_id} → Kategori Öneri Skorları")
    plt.legend()
    plt.tight_layout()
    plt.savefig("autoencoder_model/prediction_plot.png")
    plt.close()

    print("Görsel kaydedildi: prediction_plot.png")

if __name__ == "__main__":
    predict_for_customer(customer_index=0)


# Görselleştirme Açıklaması (prediction_plot.png):
# - Bu grafik, bir müşterinin farklı ürün kategorilerinde modelin tahmin ettiği satın alma skorlarını gösterir.
# - X ekseni: Ürün kategorileri
# - Y ekseni: Tahmin skoru (0 ile 1 arası)
# - Yeşil çubuklar: Modelin ilgili kategoride satın alma potansiyelinin yüksek olduğunu düşündüğü alanlar (skor ≥ 0.5)
# - Gri çubuklar: Düşük satın alma potansiyeli (skor < 0.5)
# - Kırmızı kesikli çizgi: 0.5'lik karar eşiğini gösterir
# - Bu örnekte ALFKI müşterisi için "Condiments", "Confections", "Dairy Products" ve "Produce" kategorilerinin skoru 1.0'a yakın olup önerilmiştir.
