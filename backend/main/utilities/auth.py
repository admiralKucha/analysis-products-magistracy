import json
from collections.abc import Awaitable, Callable
from functools import wraps

from cryptography.fernet import Fernet, InvalidToken
from fastapi import Response
from init import database

key = Fernet.generate_key()
cipher_suite = Fernet(key)


def login_required(func: Callable[..., Awaitable[Response]]) -> Callable[..., Awaitable[Response]]:
    @wraps(func)
    async def wrapper(*args, **kwargs: dict) -> Response:  # noqa: ANN002
        session: str | None = kwargs.get("session")
        if session is None:
            res = {"status": "error", "message": "Необходимо пройти авторизацию"}
            res = Response(content=json.dumps(res, ensure_ascii=False), status_code=401, media_type="application/json")
            res.delete_cookie("session")
            return res

        try:
            user_id = int(cipher_suite.decrypt(session))
        except InvalidToken:
            # не наш токен
            res = {"status": "error", "message": "Необходимо заново пройти авторизацию"}
            res = Response(content=json.dumps(res, ensure_ascii=False), status_code=401, media_type="application/json")
            res.delete_cookie("session")
            return res

        if await database.checker(user_id) is None:
            res = {"status": "error", "message": "Необходимо заново пройти авторизацию"}
            res = Response(content=json.dumps(res, ensure_ascii=False), status_code=401, media_type="application/json")
            res.delete_cookie("session")
            return res

        kwargs["session"] = user_id
        return await func(*args, **kwargs)

    return wrapper


def logout_check(func: Callable[..., Awaitable[Response]]) -> Callable[..., Awaitable[Response]]:
    @wraps(func)
    async def wrapper(*args, **kwargs: dict) -> Response:  # noqa: ANN002
        session: str | None = kwargs.get("session")
        if session is None:
            res = {"status": "error", "message": "Пользователь не был в аккаунте"}
            res = Response(content=json.dumps(res, ensure_ascii=False), status_code=401, media_type="application/json")
            res.delete_cookie("session")
            return res

        try:
            user_id = int(cipher_suite.decrypt(session))
        except InvalidToken:
            # не наш токен
            res = {"status": "error", "message": "Пользователь не был в аккаунте"}
            res = Response(content=json.dumps(res, ensure_ascii=False), status_code=401, media_type="application/json")
            res.delete_cookie("session")
            return res

        if await database.checker(user_id) is None:
            res = {"status": "error", "message": "Пользователь не был в аккаунте"}
            res = Response(content=json.dumps(res, ensure_ascii=False), status_code=401, media_type="application/json")
            res.delete_cookie("session")
            return res

        return await func(*args, **kwargs)

    return wrapper


def anonymous(func: Callable[..., Awaitable[Response]]) -> Callable[..., Awaitable[Response]]:
    @wraps(func)
    async def wrapper(*args, **kwargs: dict) -> Response:  # noqa: ANN002
        session: str | None = kwargs.get("session")
        if session is not None:
            try:
                _ = int(cipher_suite.decrypt(session))
            except InvalidToken:
                # не наш токен
                return await func(*args, **kwargs)

            res = {"status": "warning", "message": "Пользователь уже авторизован"}
            return Response(content=json.dumps(res, ensure_ascii=False), status_code=403, media_type="application/json")

        return await func(*args, **kwargs)

    return wrapper
