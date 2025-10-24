import json
from typing import Annotated

import init
from fastapi import APIRouter, Cookie, Query, Response
from fastapi.responses import RedirectResponse
from init import database_customer, database_products
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
                       order: str = Cookie(default=None, include_in_schema=False),
                       to_html: str = "no") -> Response:  # noqa: FAST002
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
    if to_html == "no":
        res = Response(content=json.dumps(res, ensure_ascii=False), status_code=200, media_type="application/json")
    else:
        res = RedirectResponse(url="/account", status_code=303)

    res.set_cookie("order", json.dumps({}), max_age=60 * 60 * 24 * 7)
    return res


@router.get("/order/recomendation/apriori", tags=["Покупатель"])
@login_required
async def show_recomendation_apriori(session: str = Cookie(default=None, include_in_schema=False),  # noqa: FAST002
                                     order: str = Cookie(default=None, include_in_schema=False),
                                     min_confidence: float = 0.1) -> Response:  # noqa: FAST002
    if order is None:
        order = {}

    else:
        try:
            order = json.loads(order)
        except Exception as e:
            order = {}

    recom = {}
    for el in order:
        mini_rec = init.apriori2_rules.get(int(el), {})
        for key, confidence in mini_rec.items():
            old_confidence = recom.get(key, 0)
            if confidence > old_confidence:
                recom[key] = confidence
            else:
                recom[key] = old_confidence

    print(len(init.apriori2_rules))

    res = await database_products.get_products(None, None, 100, 1, list(recom.keys()))
    res = {"status": "success", "message": "Получена рекомендации к заказу", "data": res}
    res = Response(content=json.dumps(res, ensure_ascii=False), status_code=200, media_type="application/json")
    res.set_cookie("order", json.dumps(order), max_age=60 * 60 * 24 * 7)
    return res
