import logging

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
                                                                ("WHERE orders.eval_set = 'train' AND user_id = $1 "),
                                                                 [customer_id])

                # Получили заказы
                str_exec = ("SELECT id, order_dow, order_hour_of_day "
                            "FROM orders "
                            "WHERE orders.eval_set = 'train' AND user_id = $1 "
                            "ORDER BY order_number DESC "
                            "LIMIT $2 OFFSET $3;")

                orders = await cursor.fetch(str_exec, customer_id, limit, offset)
                info = {order[0]: {"id": order[0], "order_dow": order[1], "order_hour_of_day": order[2], "products": []} for order in orders}
                orders_keys = list(info.keys())

                # Получили позиции в заказах
                str_exec = ("SELECT order_id, product_name, category_id, department_id "
                            "FROM orders_products "
                            "INNER JOIN products ON orders_products.product_id = products.id "
                            "WHERE order_id = ANY($1) "
                            "ORDER BY add_to_cart_order ASC ")
                orders_products = await cursor.fetch(str_exec, orders_keys)
                for el in orders_products:
                    info[el[0]]["products"].append({"name": el[1], "category_id": el[2], "department_id": el[3]})

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
