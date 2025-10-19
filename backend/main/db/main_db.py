import logging
import math

import asyncpg
from model.user import UserLoaded
from utilities.other import db_database, db_host, db_password, db_port, db_username


class PostgresDB:
    def __init__(self) -> None:
        self.user = db_username
        self.password = db_password
        self.host = db_host
        self.port = db_port
        self.database = db_database
        self.connection = None

        self.KEY_CHECKER = ["id", "username", "user_group", "is_banned"]
        self.KEY_STR_CHECKER = ", ".join(self.KEY_CHECKER)

    async def create_pool(self) -> None:
        self.connection = await asyncpg.create_pool(user=self.user,
                                                    password=self.password,
                                                    host=self.host,
                                                    port=self.port,
                                                    database=self.database,
                                                    min_size=5,
                                                    max_size=5,
                                                    )

    async def delete_pool(self) -> None:
        if self.connection is not None:
            await self.connection.close()

    async def checker(self, user_id: int) -> None:
        async with self.connection.acquire() as cursor:
            try:
                # Подгружаем информацию о пользователе
                str_exec = (f"SELECT {self.KEY_STR_CHECKER} FROM all_users"  # noqa: S608
                            f" WHERE id = $1")
                res_temp = await cursor.fetchrow(str_exec, user_id)

                # Пользователя нет
                if res_temp is None:
                    return None

                # Группируем информацию
                buf = dict(zip(self.KEY_CHECKER, res_temp, strict=True))
                return UserLoaded(**buf)

            except Exception as error:
                logging.error(f"checker: {error}")
                return None

    async def create_pagination(self, current_page: int, limit: int, obj: str,
                                cursor: asyncpg.Connection, condition: str = " ",
                                sql_args: list | None = None) -> tuple[int, dict]:

        if sql_args is None:
            sql_args = []

        # просчитываем offset
        offset = (current_page - 1) * limit
        # Делаем мета информацию для пользователя
        str_exec = f"SELECT COUNT(*) FROM {obj} {condition};"  # noqa: S608
        all_records = (await cursor.fetchrow(str_exec, *sql_args))[0]
        total_pages = math.ceil(all_records / limit)
        next_page = None if current_page + 1 > total_pages else current_page + 1
        prev_page = None if current_page - 1 < 1 else current_page - 1
        # Конструируем выходной json
        pagination = {
            "total_records": all_records,
            "current_page": current_page,
            "total_pages": total_pages,
            "next_page": next_page,
            "prev_page": prev_page,
        }
        return offset, pagination
