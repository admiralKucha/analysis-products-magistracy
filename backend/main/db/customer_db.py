import logging
import random

import asyncpg
from utilities.other import delete_none

from db import main_db


class PostgresDBCustomer(main_db.PostgresDB):
    def __init__(self):
        super().__init__()

    async def after_init(self) -> None:
        self.connection = await asyncpg.create_pool(user=self.user,
                                                    password=self.password,
                                                    host=self.host,
                                                    port=self.port,
                                                    database=self.database,
                                                    min_size=5,
                                                    max_size=5,
                                                    )

    async def get_old_orders(self, customer_id: int, limit: int, current_page: int) -> dict:
        # выводим все старые заказов
        error_message = "Ошибка при работе с выводом списка заказов"

        async with self.connection.acquire() as cursor:
            try:
                # узнаем информацию для пагинации
                offset, pagination = await self.create_pagination(current_page,
                                                                limit, "orders", cursor,
                                                                ("WHERE user_id = $1 "),
                                                                 [customer_id])

                # Получили заказы
                str_exec = ("SELECT id, order_dow, order_hour_of_day "
                            "FROM orders "
                            "WHERE user_id = $1 "
                            "ORDER BY id DESC "
                            "LIMIT $2 OFFSET $3;")

                orders = await cursor.fetch(str_exec, customer_id, limit, offset)
                info = {order[0]: {"id": order[0], "order_dow": order[1], "order_hour_of_day": order[2], "products": []} for order in orders}
                orders_keys = list(info.keys())

                # Получили позиции в заказах
                str_exec = ("SELECT order_id, product_name, category_id, department_name "
                            "FROM orders_products "
                            "INNER JOIN products ON orders_products.product_id = products.id "
                            "INNER JOIN departments ON department_id = departments.id "
                            "WHERE order_id = ANY($1) "
                            "ORDER BY add_to_cart_order ASC ")
                orders_products = await cursor.fetch(str_exec, orders_keys)
                for el in orders_products:
                    info[el[0]]["products"].append({"product_name": el[1],
                                                    "category_id": el[2],
                                                    "department_name": el[3],
                                                    "price": random.randint(10, 200) * 10,  # noqa: S311
                                                    "image": f"/images/{el[3]}.png"})

                res = list(info.values())

                # все хорошо
                return {"status": "success",
                    "data": res,
                    "message": "Получен список всех заказов",
                    "code": 200,
                    "pagination": pagination}

            except Exception as error:
                logging.error(f"get_old_orders: {error}")
                return {"status": "error",
                    "message": error_message,
                    "code": 500}

    async def get_info_order(self, order: dict) -> dict:
        # выводим все старые заказов
        error_message = "Ошибка при работе с выводом информации о заказе"

        async with self.connection.acquire() as cursor:
            try:
                orders_keys = [int(el) for el in order]

                # Получили позиции в заказах
                str_exec = ("SELECT products.id, product_name, category_id, department_name "
                            "FROM products "
                            "INNER JOIN departments ON department_id = departments.id "
                            "WHERE products.id = ANY($1) "
                            "ORDER BY products.id ASC ")

                orders_products = await cursor.fetch(str_exec, orders_keys)
                res = [{"id": el[0],
                         "product_name": el[1],
                         "category_id": el[2],
                         "department_name": el[3],
                         "price": random.randint(10, 200) * 10,  # noqa: S311
                         "image": f"/images/{el[3]}.png"}
                         for el in orders_products]

                # все хорошо
                return {"status": "success",
                    "data": res,
                    "message": "Получена информация по заказу",
                    "code": 200,}

            except Exception as error:
                logging.error(f"get_info_order: {error}")
                return {"status": "error",
                    "message": error_message,
                    "code": 500}

    async def create_order(self, customer_id: int, dict_order: dict):
        # создаем заказ
        error_message = "Ошибка при работе с функцией добавления заказа"

        async with self.connection.acquire() as cursor:
            try:
                transaction = cursor.transaction()
                # Начало транзакции
                await transaction.start()

                # Добавляем вакансию
                str_exec = ("INSERT INTO orders (user_id, eval_set, order_number, order_dow, order_hour_of_day)"
                            " VALUES ($1, 'prod', 100, 1, 2) RETURNING id;")
                order_id = await cursor.fetchrow(str_exec, customer_id)

                order_id = order_id[0]
                for i, order in enumerate(dict_order):
                    str_exec = ("INSERT INTO orders_products (order_id, product_id, add_to_cart_order, reordered, eval_set)"
                                " VALUES ($1, $2, $3, $4,  'prod');")
                    await cursor.execute(str_exec, order_id, int(order), i, False)

                # Сохраняем результат
                await transaction.commit()
                return {
                    "status": "success",
                    "message": "Заказ добавлен",
                    "code": 201
                }

            except Exception as error:
                logging.error(f"create_order: {error}")
                await transaction.rollback()
                return {
                    "status": "error",
                    "message": error_message,
                    "code": 500,
                }
