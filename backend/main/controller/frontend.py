import json
import random
from typing import Annotated

import init
from fastapi import APIRouter, Cookie, Query, Request
from fastapi.responses import HTMLResponse
from init import database_customer, database_products, templates
from utilities.auth import anonymous, check_session, login_required

router = APIRouter(prefix="")


@router.get("/authentication")
@anonymous
async def authentication_form(request: Request,
                              session: str = Cookie(default=None, include_in_schema=False)) -> HTMLResponse:  # noqa: FAST002
    return templates.TemplateResponse(request=request, name="auth.html")


@router.get("/")
async def main_page(request: Request,
                   current_page: Annotated[int, Query(gt=0)] = 1) -> HTMLResponse:

    res = await database_products.get_products(None, None, 16, current_page)
    page = res["pagination"]["next_page"]
    res = res["data"]

    if page % 2 == 0:
        return templates.TemplateResponse(
            request=request,
            name="products1.html",
            context={"products": res, "page": page},
        )

    return templates.TemplateResponse(
        request=request,
        name="products2.html",
        context={"products": res, "page": page},
    )


@router.get("/account")
@login_required
async def show_account(request: Request,
                       session: str = Cookie(default=None, include_in_schema=False),
                       order: str = Cookie(default=None, include_in_schema=False),
                       current_page: Annotated[int, Query(gt=0)] = 1) -> HTMLResponse:

    if order is None:
        order = {}

    else:
        try:
            order = json.loads(order)
        except Exception as e:
            order = {}

    res = await database_customer.get_info_order(order)

    return templates.TemplateResponse(
        request=request,
        name="account.html",
        context={"current_order": res["data"], "page": 1},
    )


@router.get("/old-orders")
@login_required
async def show_old_orders(request: Request,
                       session: str = Cookie(default=None, include_in_schema=False),
                       current_page: Annotated[int, Query(gt=0)] = 1) -> HTMLResponse:

    customer_id = session
    res = await database_customer.get_old_orders(customer_id, 10, current_page)
    page = res["pagination"]["next_page"]
    current_page = res["pagination"]["current_page"]
    res = res["data"]

    return templates.TemplateResponse(
        request=request,
        name="old_orders.html",
        context={"orders": res, "page": page, "current_page": current_page},
    )


@router.get("/recomendation/k_means", tags=["Покупатель"])
@check_session
async def show_recomendation_k_means_template(request: Request,
                                              session: str = Cookie(default=None, include_in_schema=False),  # noqa: FAST002
                                              page: int = 1) -> HTMLResponse:  # noqa: FAST002
    user_id = session
    if user_id is None:
        products = random.sample(init.top1000_products, min(4, len(init.top1000_products)))

    else:
        cluster_id = init.k_means_rules["users"].get(int(user_id), 0)
        products = init.k_means_rules["clusters"][cluster_id]
        products = random.sample(products, min(4, len(products)))

    res = await database_products.get_products(None, None, 4, 1, products)

    return templates.TemplateResponse(
        request=request,
        name=f"recomendation_section_{page}.html",
        context={"products": res["data"]},
    )


@router.get("/recomendation/apriori", tags=["Покупатель"])
@login_required
async def show_recomendation_apriori(request: Request,
                                     session: str = Cookie(default=None, include_in_schema=False),  # noqa: FAST002
                                     order: str = Cookie(default=None, include_in_schema=False)) -> HTMLResponse:  # noqa: FAST002
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

    res = await database_products.get_products(None, None, 100, 1, list(recom.keys()))
    return templates.TemplateResponse(
        request=request,
        name="recomendation_account.html",
        context={"current_order": res["data"]},
    )
