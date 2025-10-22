import json
from typing import Annotated

from fastapi import APIRouter, Cookie, Form, Request, Response
from init import database_guest, templates
from model.user import UserAuth
from swagger.responces.guest_responces import ResponseAuthentication, ResponseLogout
from utilities.auth import anonymous, cipher_suite, logout_check

router = APIRouter(prefix="")


@router.post("/authentication", tags=["Гость"], responses=ResponseAuthentication)
@anonymous
async def authentication(username: Annotated[str, Form()],
                         password: Annotated[str, Form()],
                         session: str = Cookie(default=None, include_in_schema=False),
                        ) -> Response:
    # проверяем модель
    user = UserAuth(
        username=username,
        password=password,
        remember_me=True,
    )

    # авторизация
    res = await database_guest.authentication_user(user.model_dump())  # узнали, есть ли человек
    code = res.pop("code")

    if res["status"] != "error":
        # пользователь прошел авторизацию
        user_id, max_age = res.pop("id"), res.pop("max_age")
        response = Response(content=json.dumps(res, ensure_ascii=False), status_code=code,
                            media_type="application/json", headers={"Accept": "application/json"})

        # добавляем cookie
        response.set_cookie("session", cipher_suite.encrypt(str(user_id).encode()).decode(), max_age=max_age)
        response.set_cookie("order", {}, max_age=max_age)
        return response

    # пользователь не прошел авторизацию, удаляем старые cookie, если есть
    res = Response(content=json.dumps(res, ensure_ascii=False), status_code=code,
                   media_type="application/json", headers={"Accept": "application/json"})
    res.delete_cookie("session")
    return res


@router.post("/logout", tags=["Гость"], responses=ResponseLogout)
@logout_check
async def logout(session: str = Cookie(default=None, include_in_schema=False)) -> Response:  # noqa: FAST002
    # выход из аккаунта
    res = {
        "status": "success",
        "message": "Выход из аккаунта совершен",
    }
    res = Response(content=json.dumps(res, ensure_ascii=False), status_code=200, media_type="application/json")
    res.delete_cookie("session")
    return res


@router.get("/authentication")
@anonymous
async def authentication_form(request: Request,
                              session: str = Cookie(default=None, include_in_schema=False)) -> Response:  # noqa: FAST002
    return templates.TemplateResponse(request=request, name="auth.html")
