import json
import math
import os
from datetime import date

# БАЗА ДАННЫХ
db_username = os.getenv("DB_USERNAME") or "admin"
db_password = os.getenv("DB_PASSWORD") or "admin"
db_host = os.getenv("DB_HOST") or "postgres-astra"
db_port = os.getenv("DB_PORT") or "5432"
db_database = os.getenv("DB_DATABASE") or "astra_career"


def delete_none(info: list[list], keys: list[str]) -> list[dict]:
    # Пробегаемся по списку, меняем полученный список на списков на список словарей
    array_res = []

    for obj in info:
        # Если в словаре были пустые значения- удаляем
        dict_res = {key: value for key, value in zip(keys, obj, strict=True) if value is not None}
        array_res.append(dict_res)

    return array_res


def delete_none_date(info: list, keys: list[str], _format: str = "%Y-%m-%d") -> dict:
    # Пробегаемся по списку - делаем его словарем, плюс переводим даты и удаляем None
    dict_res = {}
    for key, value in zip(keys, info, strict=True):
        if value is None:
            continue

        if isinstance(value, date):
            dict_res[key] = value.strftime(_format)

        else:
            dict_res[key] = value

    return dict_res


def change_date(info: list, keys: list[str], _format: str = "%Y-%m-%d") -> dict:
    # Пробегаемся по списку - делаем его словарем, плюс переводим даты
    dict_res = {}
    for key, value in zip(keys, info, strict=True):

        if isinstance(value, date):
            dict_res[key] = value.strftime(_format)

        else:
            dict_res[key] = value

    return dict_res


def delete_none_date_nested(info: list, keys: list[str], nested_keys: list[str]) -> dict:
    # Пробегаемся по списку - делаем его словарем, плюс переводим даты, плюс убираем None и
    # обрабатываем внутренюю вложенность
    dict_res = {}

    for key, value in zip(keys, info, strict=False):
        if value is None:
            continue

        if isinstance(value, date):
            dict_res[key] = value.strftime("%Y-%m-%d")
            continue

        # Обработка вложенных структур
        if key in nested_keys:
            parsed_value = json.loads(value)

            # Словари и списки обрабатываются по разному
            if isinstance(parsed_value, list):
                processed_list = [
                    {k: v for k, v in item.items() if v is not None}
                    for item in parsed_value
                    if isinstance(item, dict)
                ]
                dict_res[key] = processed_list
            else:
                dict_res[key] = parsed_value
        else:
            # Обычные значения
            dict_res[key] = value

    return dict_res


def create_pagination(all_records: int, limit: int, current_page: int) -> dict:
    # Этап пагинации
    total_pages = math.ceil(all_records / limit)
    next_page = None if current_page + 1 > total_pages else current_page + 1
    prev_page = None if current_page - 1 < 1 else current_page - 1
    return {
        "total_records": all_records,
        "current_page": current_page,
        "total_pages": total_pages,
        "next_page": next_page,
        "prev_page": prev_page,
    }
