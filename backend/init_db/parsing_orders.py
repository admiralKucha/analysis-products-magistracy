import pandas as pd
import psycopg2
import os
import numpy as np

DB_HOST = os.getenv('DB_HOST')
DB_DATABASE = os.getenv('DB_DATABASE')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT')

# Путь к CSV файлу
csv_file_path = './init_db/csv/orders.csv'

def main():
    data = pd.read_csv(csv_file_path)
    data['days_since_prior_order'] = pd.to_numeric(data['days_since_prior_order'], errors='coerce', downcast='integer')
    data['order_hour_of_day'] = pd.to_numeric(data['order_hour_of_day'], errors='coerce', downcast='integer')
    data = data.replace({np.nan: None})
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

    # Вставка всех данных одним запросом
    data_list = data[['order_id', 'user_id', 'eval_set', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']].values.tolist()

    cursor.executemany("""
    INSERT INTO orders (id, user_id, eval_set, order_number, order_dow, order_hour_of_day, days_since_prior_order) 
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, data_list)

    # Сохранение изменений и закрытие соединения
    conn.commit()
    cursor.close()
    conn.close()