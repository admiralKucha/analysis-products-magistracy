from typing import Annotated

from pydantic import BaseModel, Field, field_validator


# Информация, необходимая для авторизации
class UserAuth(BaseModel):
    username: Annotated[str, Field(min_length=4, max_length=40, examples=["test@test.ru"])]
    password: str
    remember_me: bool = False

    @field_validator("username")
    @staticmethod
    def validate_email(v: str) -> str:  # тут был cls и не было static
        return v.lower()


# Информация о юзере
class UserLoaded(BaseModel):
    id: int
    username: str
    user_group: str
    is_banned: bool
