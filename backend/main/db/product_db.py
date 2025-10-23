import logging
import random

import asyncpg

from db import main_db


class PostgresDBProduct(main_db.PostgresDB):
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

    async def get_products(self, category_id: int, department_id: int, limit: int, current_page: int) -> dict:
        # выводим все продукты по фильтры
        error_message = "Ошибка при работе с выводом списка продуктов"

        async with self.connection.acquire() as cursor:
            try:
                filter_lst = []
                filter_args = []
                arg_count = 1
                if category_id is not None:
                    filter_lst.append(f"category_id = ${arg_count}")
                    filter_args.append(category_id)
                    arg_count += 1

                if department_id is not None:
                    filter_lst.append(f"department_id = ${arg_count}")
                    filter_args.append(department_id)
                    arg_count += 1

                filter_text = "WHERE " + " AND ".join(filter_lst) if filter_lst != [] else ""

                # узнаем информацию для пагинации
                offset, pagination = await self.create_pagination(current_page,
                                                                  limit, "products", cursor,
                                                                  filter_text,
                                                                  filter_args)
                filter_args.extend((limit, offset))

                # отправляем запрос
                str_exec = ("SELECT products.id, product_name, department_name, category_name "  # noqa: S608
                            "FROM products "
                            "INNER JOIN departments ON department_id = departments.id "
                            "INNER JOIN categories ON category_id = categories.id "
                            f"{filter_text} "
                            "ORDER BY id DESC "
                            f"LIMIT ${arg_count} OFFSET ${arg_count + 1};")

                # Получаем все данные
                products = await cursor.fetch(str_exec, *filter_args)
                info = [{"id": product[0],
                         "product_name": product[1],
                         "department_name": product[2],
                         "category_name": product[2],
                         "price": random.randint(10, 200) * 10,  # noqa: S311
                         "image": f"/images/{product[2]}.png"} for product in products]

                # все хорошо
                return {"status": "success",
                    "data": info,
                    "message": "Получен список всех товаров",
                    "code": 200,
                    "pagination": pagination}

            except Exception as error:
                logging.error(f"get_products: {error}")
                return {"status": "error",
                    "message": error_message,
                    "code": 500}

    async def get_categories(self) -> dict:
        # выводим все категории
        error_message = "Ошибка при работе с выводом списка категорий"

        async with self.connection.acquire() as cursor:
            try:

                # отправляем запрос
                str_exec = ("SELECT id, category_name "
                            "FROM categories "
                            "ORDER BY id DESC ;")

                # Получаем все данные
                categories = await cursor.fetch(str_exec)
                info = [{"id": category[0],
                         "category_name": category[1]} for category in categories]

                # все хорошо
                return {"status": "success",
                    "data": info,
                    "message": "Получен список всех категорий",
                    "code": 200}

            except Exception as error:
                logging.error(f"get_categories: {error}")
                return {"status": "error",
                    "message": error_message,
                    "code": 500}

    async def get_departments(self) -> dict:
        # выводим все департаменты
        error_message = "Ошибка при работе с выводом списка департаментов"

        async with self.connection.acquire() as cursor:
            try:

                # отправляем запрос
                str_exec = ("SELECT id, department_name "
                            "FROM departments "
                            "ORDER BY id DESC ;")

                # Получаем все данные
                departments = await cursor.fetch(str_exec)
                info = [{"id": department[0],
                         "department_name": department[1]} for department in departments]

                # все хорошо
                return {"status": "success",
                    "data": info,
                    "message": "Получен список всех департаментов",
                    "code": 200}

            except Exception as error:
                logging.error(f"get_departments: {error}")
                return {"status": "error",
                    "message": error_message,
                    "code": 500}

