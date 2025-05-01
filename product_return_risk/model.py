# Model oluşturma, eğitim ve görselleştirme

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from database import get_order_data

# Sahte etiketleme: Yüksek indirim ve düşük harcama

def generate_labels(df):
    df['is_returned'] = ((df['discount'] > 0.2) & (df['total'] < 500)).astype(int)
    return df

# Model yapısı
def build_model(input_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Eğitim performans grafiği
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

if __name__ == "__main__":
    df = get_order_data()
    df = generate_labels(df)

    X = df[['discount', 'quantity', 'total']]
    y = df['is_returned']

    # class weights
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    class_weights = {i: weights[i] for i in range(len(weights))}
    print("Class Weights:", class_weights)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=20, batch_size=16, class_weight=class_weights, verbose=1)

    model.save("model.h5")
    joblib.dump(None, "scaler.pkl")  # Scaler kullanılmadı ama predict kodunda uyumluluk için kayıt
    plot_training(history)
    print("Model ve grafik kaydedildi.")
    
# discount, quantity ve total'e göre sahte etiketli iade riski tahmini yapan model eğitimi. 
# İadelerin daha önemli kabul edildiği cost-sensitive learning uygulanır. Eğitim süreci grafikle görselleştirilir.