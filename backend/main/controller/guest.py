import json
import os
from typing import Annotated

from fastapi import APIRouter, Cookie, Form, Response
from fastapi.responses import FileResponse
from init import database_guest
from model.user import UserAuth
from swagger.responces.guest_responces import ResponseAuthentication, ResponseLogout
from utilities.auth import anonymous, cipher_suite, logout_check

router = APIRouter(prefix="")
allowed_characters_image = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."

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


@router.get("/images/{image}", tags=["Гость"])
async def download_image(image: str, session: str = Cookie(default=None, include_in_schema=False)) -> FileResponse:
    # пользователь хочет получить картинку - отправляем ее

    # Оставляем только разрешенные символы в имени файла
    cleaned_filename = "".join(c if c in allowed_characters_image else "_" for c in image)

    # проверяем, есть ли картинка
    path = f"../images/img/{cleaned_filename}"
    if os.path.exists(path):
        return FileResponse(path=path)

    res = {"status": "error", "message": "Такого файла нет"}
    return Response(content=json.dumps(res, ensure_ascii=False), status_code=404, media_type="application/json")
