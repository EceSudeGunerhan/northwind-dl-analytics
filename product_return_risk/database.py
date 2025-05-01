# Veriyi SQL üzerinden çeken fonksiyon

from sqlalchemy import create_engine
import pandas as pd

def get_order_data():
    engine = create_engine("postgresql://postgres:1234@localhost:5432/gyk2Northwind")

    query = """
    SELECT 
        order_id,
        product_id,
        quantity,
        unit_price,
        discount,
        quantity * unit_price * (1 - discount) AS total
    FROM order_details
    """

    df = pd.read_sql_query(query, engine)
    return df
