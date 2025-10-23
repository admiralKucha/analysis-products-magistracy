from typing import Annotated

from fastapi import APIRouter, Cookie, Query, Request
from fastapi.responses import HTMLResponse
from init import database_customer, database_products, templates
from utilities.auth import anonymous, login_required

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
                       current_page: Annotated[int, Query(gt=0)] = 1) -> HTMLResponse:

    customer_id = session
    res = await database_customer.get_old_orders(customer_id, 10, current_page)
    page = res["pagination"]["next_page"]
    res = res["data"]

    return templates.TemplateResponse(
        request=request,
        name="account.html",
        context={"orders": res, "page": page},
    )
