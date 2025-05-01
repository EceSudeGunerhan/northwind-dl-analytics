# model oluşturma ve eğitme.
# veri hazırlama + hedef değişken + eğitim

# Bu dosya, oluşturulan müşteri özet verileriyle derin öğrenme modeli eğitir ve scaler'ı kaydeder.
# Model başarıyla eğitildikten sonra hem 'model.h5' hem 'scaler.pkl' dosyaları oluşturulur.

# Model eğitimi dosyası: temporal feature, veri çoğaltma, SMOTE, görselleştirme içerir.

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from database import get_order_data_for_labeling

def extract_temporal_features(df):
    df['last_order_date'] = pd.to_datetime(df['last_order_date'])
    df['order_month'] = df['last_order_date'].dt.month
    df['order_season'] = df['order_month'].apply(lambda x: (
        1 if x in [12, 1, 2] else
        2 if x in [3, 4, 5] else
        3 if x in [6, 7, 8] else
        4))
    return df

def augment_data(df, multiplier=1):
    augmented = []
    for _ in range(multiplier):
        temp = df.copy()
        temp['total_orders'] *= np.random.normal(1.0, 0.05, len(temp))
        temp['total_spent'] *= np.random.normal(1.0, 0.05, len(temp))
        temp['avg_order_value'] *= np.random.normal(1.0, 0.05, len(temp))
        augmented.append(temp)
    return pd.concat([df] + augmented, ignore_index=True)

def plot_class_distribution(y_before, y_after):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    pd.Series(y_before).value_counts().plot(kind="bar", ax=axs[0], title="Before SMOTE")
    pd.Series(y_after).value_counts().plot(kind="bar", ax=axs[1], title="After SMOTE")
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.close()

def plot_training(history):
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['loss'], label='Loss')
    plt.title("Model Training Performance")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.close()

def plot_seasonal_trend(df):
    season_map = {1: "Kış", 2: "İlkbahar", 3: "Yaz", 4: "Sonbahar"}
    df['season_name'] = df['order_season'].map(season_map)
    seasonal_stats = df.groupby('season_name')['label'].mean()
    plt.figure(figsize=(6, 4))
    seasonal_stats.plot(kind='bar', color='skyblue')
    plt.title("Mevsime Göre Sipariş Verme Oranı (label=1)")
    plt.ylabel("Oran")
    plt.xlabel("Mevsim")
    plt.tight_layout()
    plt.savefig("seasonal_distribution.png")
    plt.close()

def build_model(input_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    df = get_order_data_for_labeling("1997-06-01")
    df = extract_temporal_features(df)
    df = augment_data(df, multiplier=2)

    features = ['total_orders', 'total_spent', 'avg_order_value', 'order_month', 'order_season']
    X = df[features]
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    plot_class_distribution(y, y_resampled)
    plot_seasonal_trend(df)

    model = build_model(X_resampled.shape[1])
    history = model.fit(X_resampled, y_resampled, epochs=20, batch_size=16, verbose=1)
    model.save("model.h5")
    plot_training(history)
    print("Model ve görseller kaydedildi.")

# Bu dosya, oluşturulan müşteri özet verileriyle derin öğrenme modeli eğitir.
# Başta yalnızca label=0 verisi bulunduğu için model eğitilemiyordu.
# Daha uygun bir tarih kesiti seçilerek (1997-06-01), hem 0 hem de 1 içeren dengeli veri elde edildi.
# SMOTE yöntemi ile azınlık sınıfı çoğaltılarak veri dengesi sağlandı.
# Model başarıyla eğitilip 'model.h5' olarak kaydedildi.
