# SQLAlchemy ile veri çekimi yapar.

# Bu dosya, geçmiş ve gelecek siparişlere göre müşteri davranışını analiz eder.
# cutoff_date = "1997-06-01" olarak belirlendi; bu tarih veritabanı kapsamına uygun.
# Gelecek 6 ayda sipariş varsa label = 1, yoksa label = 0 olarak atanır.

# Veritabanından müşteri sipariş geçmişini getirir ve 6 ay içinde tekrar sipariş verip vermediğini label olarak belirler.

from sqlalchemy import create_engine
import pandas as pd

def get_order_data_for_labeling(cutoff_date="1997-06-01"):
    engine = create_engine("postgresql://postgres:1234@localhost:5432/gyk2Northwind")

    query = f"""
    WITH history AS (
        SELECT c.customer_id,
               COUNT(o.order_id) AS total_orders,
               SUM(od.unit_price * od.quantity) AS total_spent,
               AVG(od.unit_price * od.quantity) AS avg_order_value,
               MAX(o.order_date) AS last_order_date
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN order_details od ON o.order_id = od.order_id
        WHERE o.order_date < DATE '{cutoff_date}'
        GROUP BY c.customer_id
    ),
    future AS (
        SELECT customer_id,
               COUNT(*) AS future_orders
        FROM orders
        WHERE order_date >= DATE '{cutoff_date}'
          AND order_date < (DATE '{cutoff_date}' + INTERVAL '6 months')
        GROUP BY customer_id
    )
    SELECT h.*,
           COALESCE(f.future_orders, 0) AS future_orders,
           CASE WHEN COALESCE(f.future_orders, 0) > 0 THEN 1 ELSE 0 END AS label
    FROM history h
    LEFT JOIN future f ON h.customer_id = f.customer_id
    """

    df = pd.read_sql_query(query, engine)
    return df


# Bu dosya, geçmiş ve gelecek siparişlere göre müşteri davranışını analiz eder.
# Başlangıçta kullanılan cutoff_date ("2007-01-01") veritabanı kapsamı dışındaydı ve
# bu nedenle tüm müşterilerin label değeri 0 oluyordu.
# Sorun, cutoff_date değerini "1997-06-01" olarak değiştirerek çözüldü.
# Bu sayede bazı müşteriler için gelecekte sipariş olduğu tespit edilip label=1 atanabildi.
