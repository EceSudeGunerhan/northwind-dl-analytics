# Müşteri-kategori etkileşim verisi (pozitif ve negatif örnekler dahil)

import pandas as pd
from sqlalchemy import create_engine
import random

def get_user_category_interactions():
    engine = create_engine("postgresql://postgres:1234@localhost:5432/gyk2Northwind")

    # Her müşteri hangi kategoriden alışveriş yapmış
    query = """
    SELECT DISTINCT
        c.customer_id,
        cat.category_id
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_details od ON o.order_id = od.order_id
    JOIN products p ON od.product_id = p.product_id
    JOIN categories cat ON p.category_id = cat.category_id
    """

    df = pd.read_sql_query(query, engine)

    # Kullanıcı ve kategori listeleri
    customers = df['customer_id'].unique().tolist()
    categories = df['category_id'].unique().tolist()

    # Pozitif örnekler (etkileşim var)
    df['label'] = 1
    positive_samples = df.copy()

    # Negatif örnekler (etkileşim olmayan kombinasyonlardan rastgele seçilir)
    negative_samples = []
    for customer in customers:
        interacted = set(positive_samples[positive_samples['customer_id'] == customer]['category_id'])
        not_interacted = set(categories) - interacted
        for neg_cat in random.sample(list(not_interacted), min(2, len(not_interacted))):  # 2 negatif örnek al
            negative_samples.append({'customer_id': customer, 'category_id': neg_cat, 'label': 0})

    negative_df = pd.DataFrame(negative_samples)

    # Pozitif + Negatif birleştir
    interactions = pd.concat([positive_samples, negative_df], ignore_index=True)
    return interactions

# Bu dosya, NCF (Neural Collaborative Filtering) modeli için kullanıcı-kategori etkileşim verisi üretir.
# - Pozitif örnekler: Müşterinin gerçekten alışveriş yaptığı kategori eşleşmeleri (label = 1)
# - Negatif örnekler: Müşterinin alışveriş yapmadığı kategorilerden rastgele seçilen eşleşmeler (label = 0)
# - Veri formatı: (customer_id, category_id, label)
# Bu yapı, embedding tabanlı öneri sistemleri için uygun bir etkileşim veri seti oluşturur.
