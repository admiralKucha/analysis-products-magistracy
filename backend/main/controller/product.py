import json
from typing import Annotated

from fastapi import APIRouter, Cookie, Query, Response
from init import database_products

router = APIRouter(prefix="/api/products")


@router.get("/", tags=["Проудукты"])
async def show_products(session: str = Cookie(default=None, include_in_schema=False),
                        category_id: int | None = None, department_id: int | None = None,
                        limit: Annotated[int, Query(gt=0)] = 10, current_page: Annotated[int, Query(gt=0)] = 1) -> None:

    # получаем всю информацию о продуктах
    res = await database_products.get_products(category_id, department_id, limit, current_page)
    code = res.pop("code")
    if res["status"] == "error":
        return Response(content=json.dumps(res, ensure_ascii=False), status_code=code, media_type="application/json")

    return Response(content=json.dumps(res, ensure_ascii=False), status_code=code, media_type="application/json")


@router.get("/categories", tags=["Проудукты"])
async def show_categories(session: str = Cookie(default=None, include_in_schema=False)) -> None:

    # получаем всю информацию о категориях
    res = await database_products.get_categories()
    code = res.pop("code")
    if res["status"] == "error":
        return Response(content=json.dumps(res, ensure_ascii=False), status_code=code, media_type="application/json")

    return Response(content=json.dumps(res, ensure_ascii=False), status_code=code, media_type="application/json")


@router.get("/departments", tags=["Проудукты"])
async def show_departments(session: str = Cookie(default=None, include_in_schema=False)) -> None:

    # получаем всю информацию о департаментах
    res = await database_products.get_departments()
    code = res.pop("code")
    if res["status"] == "error":
        return Response(content=json.dumps(res, ensure_ascii=False), status_code=code, media_type="application/json")

    return Response(content=json.dumps(res, ensure_ascii=False), status_code=code, media_type="application/json")

