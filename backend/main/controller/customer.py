import json
from typing import Annotated

from fastapi import APIRouter, Cookie, Query, Response
from init import database_customer
from utilities.auth import login_required

router = APIRouter(prefix="/api/customer")


@router.get("/old/orders", tags=["Покупатель"])
@login_required
async def show_old_orders(session: str = Cookie(default=None, include_in_schema=False),
                          limit: Annotated[int, Query(gt=0)] = 10, current_page: Annotated[int, Query(gt=0)] = 1) -> None:
    customer_id = session

    # получаем всю информацию о прошлых покупках
    res = await database_customer.get_old_orders(customer_id, limit, current_page)
    code = res.pop("code")
    if res["status"] == "error":
        return Response(content=json.dumps(res, ensure_ascii=False), status_code=code, media_type="application/json")

    return Response(content=json.dumps(res, ensure_ascii=False), status_code=code, media_type="application/json")
