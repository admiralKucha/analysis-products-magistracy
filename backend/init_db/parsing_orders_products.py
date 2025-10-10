import pandas as pd
import psycopg2
import os

DB_HOST = os.getenv('DB_HOST') or "localhost"
DB_DATABASE = os.getenv('DB_DATABASE') or "products_db"
DB_USERNAME = os.getenv('DB_USERNAME') or "postgres"
DB_PASSWORD = os.getenv('DB_PASSWORD') or "postgres"
DB_PORT = os.getenv('DB_PORT') or 5433


def main(name):
    data = pd.read_csv(f'./init_db/csv/order_products__{name}.csv')
    data['order_id'] = data['order_id'].astype(int)
    data['product_id'] = data['product_id'].astype(int)
    data['add_to_cart_order'] = data['add_to_cart_order'].astype(int)
    data['reordered'] = data['reordered'].astype(bool)

    # Установка соединения с базой данных
    conn = psycopg2.connect(
        dbname=DB_DATABASE,
        user=DB_USERNAME,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()
    # Вставка данных в таблицу
    for _, row in data.iterrows():
        cursor.execute("""
        INSERT INTO orders_products (order_id, product_id, add_to_cart_order, reordered, eval_set) VALUES (%s, %s, %s, %s, %s)
        """, (
            row['order_id'], 
            row['product_id'], 
            row['add_to_cart_order'], 
            row['reordered'], 
            name
        ))

    # Сохранение изменений и закрытие соединения
    conn.commit()
    cursor.close()
    conn.close()
