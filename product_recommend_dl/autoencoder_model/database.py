# Her müşterinin kategori bazlı toplam harcamasını döndüren fonksiyon

from sqlalchemy import create_engine
import pandas as pd

def get_customer_category_matrix():
    engine = create_engine("postgresql://postgres:1234@localhost:5432/gyk2Northwind")

    query = """
    SELECT 
        c.customer_id,
        cat.category_name,
        SUM(od.unit_price * od.quantity) AS total_spent
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_details od ON o.order_id = od.order_id
    JOIN products p ON od.product_id = p.product_id
    JOIN categories cat ON p.category_id = cat.category_id
    GROUP BY c.customer_id, cat.category_name
    """

    df = pd.read_sql_query(query, engine)

    # Müşteri × Kategori harcama matrisi
    pivot = df.pivot_table(index='customer_id', columns='category_name', values='total_spent', fill_value=0)

    return pivot.reset_index()
