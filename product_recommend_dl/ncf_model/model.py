# Kullanıcı-kategori etkileşim verisine dayalı Neural Collaborative Filtering (NCF) modeli eğitimi

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from database import get_user_category_interactions
import matplotlib.pyplot as plt

def build_ncf_model(n_users, n_categories, latent_dim=16):
    user_input = tf.keras.Input(shape=(1,), name='user_input')
    category_input = tf.keras.Input(shape=(1,), name='category_input')

    user_embed = tf.keras.layers.Embedding(n_users, latent_dim)(user_input)
    category_embed = tf.keras.layers.Embedding(n_categories, latent_dim)(category_input)

    user_vec = tf.keras.layers.Flatten()(user_embed)
    category_vec = tf.keras.layers.Flatten()(category_embed)

    x = tf.keras.layers.Concatenate()([user_vec, category_vec])
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[user_input, category_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    df = get_user_category_interactions()

    # Label encoding
    user_encoder = LabelEncoder()
    cat_encoder = LabelEncoder()

    df['user_id_enc'] = user_encoder.fit_transform(df['customer_id'])
    df['cat_id_enc'] = cat_encoder.fit_transform(df['category_id'])

    X_user = df['user_id_enc'].values
    X_cat = df['cat_id_enc'].values
    y = df['label'].values

    X_train_u, X_test_u, X_train_c, X_test_c, y_train, y_test = train_test_split(
        X_user, X_cat, y, test_size=0.2, random_state=42
    )

    n_users = df['user_id_enc'].nunique()
    n_cats = df['cat_id_enc'].nunique()

    model = build_ncf_model(n_users, n_cats)

    history = model.fit(
        [X_train_u, X_train_c], y_train,
        validation_data=([X_test_u, X_test_c], y_test),
        epochs=10, batch_size=16, verbose=1
    )

    # Modeli ve encoderları kaydet
    model.save("ncf_model/model.h5")
    np.save("ncf_model/user_encoder.npy", user_encoder.classes_)
    np.save("ncf_model/cat_encoder.npy", cat_encoder.classes_)

    # Eğitim grafiği
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title("NCF Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ncf_model/training_plot.png")
    plt.close()
    print("Model ve grafik kaydedildi.")


# Bu dosya, Neural Collaborative Filtering (NCF) mimarisiyle bir öneri sistemi modeli eğitir.
# - Girdi: (user_id, category_id) → her satır bir etkileşim (alışveriş yapmış ya da yapmamış)
# - Çıktı: binary label → 1: etkileşim var, 0: yok
# - Kullanıcı ve kategori ID'leri LabelEncoder ile sayıya çevrilir
# - NCF modeli Embedding + Dense katmanlarıyla kurulur
# - Eğitim tamamlandıktan sonra:
#    - model.h5: eğitilmiş model dosyası
#    - user_encoder.npy & cat_encoder.npy: ID çevirme için
#    - training_plot.png: eğitim ve doğrulama doğruluk görseli oluşturulur
