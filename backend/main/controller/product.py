import json
import logging
from typing import Annotated

from fastapi import APIRouter, Cookie, Query, Response
from init import database_products

router = APIRouter(prefix="/api/products")


@router.get("/", tags=["Продукты"])
async def show_products(session: str = Cookie(default=None, include_in_schema=False),
                        category_id: int | None = None, department_id: int | None = None,
                        limit: Annotated[int, Query(gt=0)] = 10, current_page: Annotated[int, Query(gt=0)] = 1) -> Response:

    # получаем всю информацию о продуктах
    res = await database_products.get_products(category_id, department_id, limit, current_page)
    code = res.pop("code")
    if res["status"] == "error":
        return Response(content=json.dumps(res, ensure_ascii=False), status_code=code, media_type="application/json")

    return Response(content=json.dumps(res, ensure_ascii=False), status_code=code, media_type="application/json")


@router.get("/categories", tags=["Продукты"])
async def show_categories(session: str = Cookie(default=None, include_in_schema=False)) -> Response:

    # получаем всю информацию о категориях
    res = await database_products.get_categories()
    code = res.pop("code")
    if res["status"] == "error":
        return Response(content=json.dumps(res, ensure_ascii=False), status_code=code, media_type="application/json")

    return Response(content=json.dumps(res, ensure_ascii=False), status_code=code, media_type="application/json")


@router.get("/departments", tags=["Продукты"])
async def show_departments(session: str = Cookie(default=None, include_in_schema=False)) -> Response:

    # получаем всю информацию о департаментах
    res = await database_products.get_departments()
    code = res.pop("code")
    if res["status"] == "error":
        return Response(content=json.dumps(res, ensure_ascii=False), status_code=code, media_type="application/json")

    return Response(content=json.dumps(res, ensure_ascii=False), status_code=code, media_type="application/json")


@router.post("/{product_id}", tags=["Продукты"])
async def add_to_order(product_id: int,
                        session: str = Cookie(default=None, include_in_schema=False),
                        order: str = Cookie(default=None, include_in_schema=False)) -> Response:

    if order is None:
        order = {}
    else:
        try:
            order = json.loads(order)
        except json.JSONDecodeError as e:
            logging.error(f"bucket: {e}")
            order = {}

    count = order.get(str(product_id), 0)
    order[str(product_id)] = count + 1

    # Подготовка ответа
    res = {"status": "success", "message": "Товар добавлен в корзину"}
    response = Response(content=json.dumps(res, ensure_ascii=False), status_code=200,
                        media_type="application/json")

    response.set_cookie("order", json.dumps(order), max_age=60 * 60 * 24 * 7)
    return response


@router.delete("/{product_id}", tags=["Продукты"])
async def delete_to_order(product_id: int,
                          session: str = Cookie(default=None, include_in_schema=False),
                          order: str = Cookie(default=None, include_in_schema=False)) -> Response:

    if order is None:
        order = {}
    else:
        try:
            order = json.loads(order)
        except json.JSONDecodeError as e:
            logging.error(f"bucket: {e}")
            order = {}

    order.pop(str(product_id), 0)

    # Подготовка ответа
    res = {"status": "success", "message": "Товар удален из корзины"}
    response = Response(content=json.dumps(res, ensure_ascii=False), status_code=200,
                        media_type="application/json")

    response.set_cookie("order", json.dumps(order), max_age=60 * 60 * 24 * 7)
    return response
