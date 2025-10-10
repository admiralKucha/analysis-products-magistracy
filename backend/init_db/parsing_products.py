import pandas as pd
import psycopg2
import os

DB_HOST = os.getenv('DB_HOST')
DB_DATABASE = os.getenv('DB_DATABASE')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT')

# Путь к CSV файлу
csv_file_path = './init_db/csv/products.csv'

def main():
    data = pd.read_csv(csv_file_path)
    print(data.dtypes)
    
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
        INSERT INTO products (id, product_name, category_id, department_id) VALUES (%s, %s, %s, %s)
        """, (row['product_id'], row['product_name'], row['aisle_id'], row['department_id']))

    # Сохранение изменений и закрытие соединения
    conn.commit()
    cursor.close()
    conn.close()