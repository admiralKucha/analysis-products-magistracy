from typing import Annotated

from pydantic import BaseModel, Field
from swagger.models.status_model import StatusEnum


class Response500(BaseModel):
    status: Annotated[StatusEnum, Field(examples=["error"])]
    message: Annotated[str, Field(examples=["Неизвестная ошибка на стороне сервера"])]


class Response422(BaseModel):
    status: Annotated[StatusEnum, Field(examples=["error"])]
    message: Annotated[str, Field(examples=["Ошибка валидации"])]
    detail: Annotated[list[dict], Field(examples=[[{"loc": ["Поле ошибки", "Вложенное поле ошибки или индекс"],
                                                    "type": "Короткое пояснение к ошибке"}]])]


################################################
################################################

class ResponseAuthentication404(BaseModel):
    status: Annotated[StatusEnum, Field(examples=["error"])]
    message: Annotated[str, Field(examples=["Такого логина не существует или пользователь заблокирован"])]


class ResponseAuthentication403(BaseModel):
    status: Annotated[StatusEnum, Field(examples=["error или warning"])]
    message: Annotated[str, Field(examples=["Неверный пароль или пользователь уже авторизован"])]


class ResponseAuthentication200(BaseModel):
    status: Annotated[StatusEnum, Field(examples=["success"])]
    message: Annotated[str, Field(examples=["Пользователь авторизован"])]


ResponseAuthentication = {200: {"model": ResponseAuthentication200},
                          403: {"model": ResponseAuthentication403}, 404: {"model": ResponseAuthentication404},
                          422: {"model": Response422},
                          500: {"model": Response500}}


################################################
################################################


class ResponseLogout200(BaseModel):
    status: Annotated[StatusEnum, Field(examples=["success"])]
    message: Annotated[str, Field(examples=["Выход из аккаунта совершен"])]


class ResponseLogout401(BaseModel):
    status: Annotated[StatusEnum, Field(examples=["error"])]
    message: Annotated[str, Field(examples=["Пользователь не был в аккаунте"])]


ResponseLogout = {200: {"model": ResponseLogout200},
                  401: {"model": ResponseLogout401},
                  422: {"model": None}}


################################################
################################################
