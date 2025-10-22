import json
from typing import Annotated

from fastapi import APIRouter, Cookie, Query, Response
from init import database_customer
from utilities.auth import login_required

router = APIRouter(prefix="/api/customer")


@router.get("/old/orders", tags=["Покупатель"])
@login_required
async def show_old_orders(session: str = Cookie(default=None, include_in_schema=False),
                          limit: Annotated[int, Query(gt=0)] = 10, current_page: Annotated[int, Query(gt=0)] = 1) -> Response:
    customer_id = session

    # получаем всю информацию о прошлых покупках
    res = await database_customer.get_old_orders(customer_id, limit, current_page)
    code = res.pop("code")
    if res["status"] == "error":
        return Response(content=json.dumps(res, ensure_ascii=False), status_code=code, media_type="application/json")

    return Response(content=json.dumps(res, ensure_ascii=False), status_code=code, media_type="application/json")


@router.get("/order", tags=["Покупатель"])
@login_required
async def show_order(session: str = Cookie(default=None, include_in_schema=False),  # noqa: FAST002
                     order: str = Cookie(default=None, include_in_schema=False)) -> Response:  # noqa: FAST002
    if order is None:
        order = {}

    else:
        try:
            order = json.loads(order)
        except Exception as e:
            order = {}
            res = {"status": "success", "message": "Получена информация по заказу", "data": order}
            res = Response(content=json.dumps(res, ensure_ascii=False), status_code=200, media_type="application/json")
            res.set_cookie("order", json.dumps(order), max_age=60 * 60 * 24 * 7)
            return res

        res = await database_customer.get_info_order(order)
        if res["status"] == "error":
            res = Response(content=json.dumps(res, ensure_ascii=False), status_code=500, media_type="application/json")

    res = {"status": "success", "message": "Получена информация по заказу", "data": res["data"]}
    res = Response(content=json.dumps(res, ensure_ascii=False), status_code=200, media_type="application/json")
    res.set_cookie("order", json.dumps(order), max_age=60 * 60 * 24 * 7)
    return res


@router.post("/order", tags=["Покупатель"])
@login_required
async def create_order(session: str = Cookie(default=None, include_in_schema=False),  # noqa: FAST002
                       order: str = Cookie(default=None, include_in_schema=False)) -> Response:  # noqa: FAST002
    customer_id = session

    if order is None:
        order = {}
        res = {"status": "error", "message": "Корзина пуста"}
        res = Response(content=json.dumps(res, ensure_ascii=False), status_code=200, media_type="application/json")
        res.set_cookie("order", json.dumps(order), max_age=60 * 60 * 24 * 7)
        return res

    try:
        order = json.loads(order)
    except Exception as e:
        order = {}
        res = {"status": "error", "message": "Корзина пуста"}
        res = Response(content=json.dumps(res, ensure_ascii=False), status_code=404, media_type="application/json")
        res.set_cookie("order", json.dumps(order), max_age=60 * 60 * 24 * 7)
        return res

    if order == {}:
        order = {}
        res = {"status": "error", "message": "Корзина пуста"}
        res = Response(content=json.dumps(res, ensure_ascii=False), status_code=200, media_type="application/json")
        res.set_cookie("order", json.dumps(order), max_age=60 * 60 * 24 * 7)
        return res

    res = await database_customer.create_order(customer_id, order)
    if res["status"] == "error":
        res = Response(content=json.dumps(res, ensure_ascii=False), status_code=500, media_type="application/json")

    res = {"status": "success", "message": "Заказ создан"}
    res = Response(content=json.dumps(res, ensure_ascii=False), status_code=200, media_type="application/json")
    res.set_cookie("order", json.dumps(order), max_age=60 * 60 * 24 * 7)
    return res
