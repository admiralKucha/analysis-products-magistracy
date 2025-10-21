import logging

import bcrypt

from db import main_db


class PostgresDBGuest(main_db.PostgresDB):
    def __init__(self) -> None:
        super().__init__()
        self.KEY_AUTH = ["id", "username", "password", "user_group", "is_banned"]
        self.KEY_STR_AUTH = ", ".join(self.KEY_AUTH)

    async def authentication_user(self, user: dict) -> dict:
        error_message = "Ошибка при работе с функцией входа пользователя в учетную запись"
        username, password, remember_me = user["username"], user["password"], user["remember_me"]

        # Нужно ли запоминать пользователя. Максимум - год
        max_age = 60 * 60 * 24 * 356 if remember_me else None

        async with self.connection.acquire() as cursor:
            try:
                # Берем информацию по логину
                str_exec = (f"SELECT {self.KEY_STR_AUTH} FROM all_users"  # noqa: S608
                            f" WHERE username = $1")
                res_temp = await cursor.fetchrow(str_exec, username)

                # Логина нет
                if res_temp is None:
                    return {
                        "status": "error",
                        "message": "Такого логина не существует",
                        "code": 404,
                    }

                # Группируем информацию
                buf = dict(zip(self.KEY_AUTH, res_temp, strict=True))
                real_password = buf.pop("password")

                # Проверяем пароль
                if bcrypt.checkpw(password.encode("utf-8"), real_password.encode("utf-8")):

                    # Заблокирован ли пользователь
                    if buf["is_banned"] is True:
                        return {
                            "status": "error",
                            "message": "Пользователь заблокирован",
                            "code": 404,
                        }

                else:
                    # Пароль неверный
                    return {
                        "status": "error",
                        "message": "Неверный пароль",
                        "code": 403,
                    }

                # если все хорошо
                return {
                    "status": "success",
                    "message": "Пользователь авторизован",
                    "id": buf["id"],
                    "max_age": max_age,
                    "user_group": buf["user_group"],
                    "code": 200,
                }

            except Exception as error:
                logging.error(f"authentication_user: {error}")
                return {
                    "status": "error",
                    "message": error_message,
                    "code": 500,
                }