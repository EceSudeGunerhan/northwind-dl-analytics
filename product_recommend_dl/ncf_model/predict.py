# NCF modeliyle önerilen kategorileri terminale yazdırır ve grafiğini çizer

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from database import get_user_category_interactions

# Model ve encoder'ları yükle
model = tf.keras.models.load_model("ncf_model/model.h5")
user_classes = np.load("ncf_model/user_encoder.npy", allow_pickle=True)
cat_classes = np.load("ncf_model/cat_encoder.npy", allow_pickle=True)

def recommend_categories_for_user(user_id):
    if user_id not in user_classes:
        print(f"Kullanıcı {user_id} eğitim verisinde yok.")
        return

    # Sayısal indekslere çevir
    user_index = np.where(user_classes == user_id)[0][0]
    cat_indices = np.arange(len(cat_classes))

    # Tahmin yap
    user_input = np.full(shape=len(cat_indices), fill_value=user_index)
    preds = model.predict([user_input, cat_indices], verbose=0).flatten()

    # En yüksek 3 tahmin
    top_indices = preds.argsort()[-3:][::-1]
    print(f"Kullanıcı: {user_id}")
    print("Önerilen Kategoriler:")
    for idx in top_indices:
        print(f" - {cat_classes[idx]} (skor: {round(float(preds[idx]), 4)})")

    # Görselleştirme
    plt.figure(figsize=(10, 5))
    plt.bar(cat_classes, preds, color=["green" if i in top_indices else "gray" for i in range(len(preds))])
    plt.xticks(rotation=45)
    plt.title(f"Kategori Skorları – Kullanıcı: {user_id}")
    plt.ylabel("Tahmin Skoru")
    plt.axhline(y=0.5, color="red", linestyle="--", label="Öneri Eşiği (0.5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ncf_model/prediction_plot.png")
    plt.close()
    print("prediction_plot.png kaydedildi.")

if __name__ == "__main__":
    df = get_user_category_interactions()
    sample_user = df['customer_id'].iloc[0]
    recommend_categories_for_user(sample_user)


# Bu dosya, eğitilmiş NCF modelini kullanarak bir kullanıcı için kategori önerileri üretir.
# - Girdi: Kullanıcının customer_id değeri (örnek: ALFKI)
# - Model: Bu kullanıcı için tüm kategorilerde satın alma olasılığı tahmini yapar
# - Çıktı:
#     - Terminalde en yüksek skora sahip ilk 3 kategori yazdırılır
#     - Tahmin edilen tüm kategori skorları görselleştirilir ve 'prediction_plot.png' olarak kaydedilir

# Görselleştirme Açıklaması:
# - X ekseni: Ürün kategorileri
# - Y ekseni: Modelin tahmin ettiği satın alma skoru (0 ile 1 arasında)
# - Yeşil çubuklar: Modelin en çok önerdiği ilk 3 kategori
# - Gri çubuklar: Diğer kategoriler
# - Kırmızı kesikli çizgi: 0.5 eşik değeri (bu değerin üzeri, satın alma potansiyeli yüksek olarak değerlendirilir)
# - Örnek çıktı:
#     Kullanıcı: ALFKI
#     Önerilen Kategoriler:
#     - Dairy Products (skor: 0.9112)
#     - Confections (skor: 0.7824)
#     - Beverages (skor: 0.6741)
