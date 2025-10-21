from typing import Annotated

from pydantic import BaseModel, Field, field_validator


# Информация, необходимая для авторизации
class UserAuth(BaseModel):
    username: Annotated[str, Field(min_length=4, max_length=40, examples=["test@test.ru"])]
    password: str
    remember_me: bool = False


# Информация о юзере
class UserLoaded(BaseModel):
    id: int
    username: str
    user_group: str
    is_banned: bool
