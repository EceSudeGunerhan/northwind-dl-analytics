# SHAP ile açıklama üretimi (XAI)

# SHAP ile modelin bir siparişi neden riskli bulduğunu açıklayan bar grafik oluşturur (scaler kullanılmaz)

import shap
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Özellik isimleri modelle uyumlu olmalı
FEATURE_NAMES = ['discount', 'quantity', 'total']

# Örnek sipariş verisi
input_data = pd.DataFrame([[0.25, 5, 300.0]], columns=FEATURE_NAMES)

# Modeli yükle
model = tf.keras.models.load_model("model.h5")

# SHAP için predict fonksiyonu — NumPy array döndürür!
def predict_fn(X):
    return model.predict(X).flatten()

# SHAP KernelExplainer oluştur
explainer = shap.KernelExplainer(predict_fn, input_data)
shap_values = explainer.shap_values(input_data)

# SHAP değerlerini kontrol et
print("SHAP values:", shap_values)
print("SHAP shape:", np.array(shap_values).shape)

# Tek gözlem için bar grafik çiz
shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values

plt.figure(figsize=(6, 4))
plt.barh(FEATURE_NAMES, shap_vals[0].flatten(), color='royalblue')
plt.xlabel("SHAP Değeri (model çıktısına etkisi)")
plt.title("SHAP Açıklaması (Tek Sipariş)")
plt.tight_layout()
plt.savefig("shap_kernel_bar.png")
plt.close()

print("SHAP bar grafiği 'shap_kernel_bar.png' olarak başarıyla kaydedildi.")


# NOT: SHAP grafiğinde tüm özelliklerin etkisi ≈ 0 görünmektedir. Bu durumun olası nedenleri:
# 1. Modelin kararları çok basit bir kurala dayanıyor olabilir (örneğin sadece 'discount' > 0.2 gibi).
# 2. Kullanılan sipariş örneği, model için çok belirgin bir sınıfa ait olabilir → SHAP fark yaratamaz.
# 3. SHAP KernelExplainer, Keras modeliyle sınırlı çalışmış olabilir (uyumsuz çıktı şekli vs.).
# 4. Model karmaşıklığı ve veri çeşitliliği düşük olduğu için SHAP etki dağılımı görünmez hale gelmiş olabilir.
# Bu nedenle SHAP açıklamaları anlamlı bir şekilde üretilememiştir.
