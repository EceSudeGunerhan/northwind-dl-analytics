# Kategori bazlı harcamalara dayalı multi-label derin öğrenme modeli eğitimi

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from database import get_customer_category_matrix

def generate_labels(X, top_k=3):
    X_copy = X.copy()
    customer_ids = X_copy.pop("customer_id")
    label_matrix = (X_copy.apply(lambda row: row >= row.nlargest(top_k).min(), axis=1)).astype(int)
    return customer_ids, X_copy, label_matrix

def train_model(X, y):
    input_dim = X.shape[1]

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(input_dim, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X, y, epochs=20, batch_size=8, validation_split=0.2, verbose=1)

    model.save("autoencoder_model/model.h5")
    return history

def plot_training(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("autoencoder_model/training_plot.png")
    plt.close()

if __name__ == "__main__":
    df = get_customer_category_matrix()
    customer_ids, X, y = generate_labels(df)
    X = X.values
    y = y.values

    history = train_model(X, y)
    plot_training(history)
    print("Model ve grafik kaydedildi.")
