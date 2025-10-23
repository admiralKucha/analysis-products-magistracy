import json
from collections.abc import Awaitable, Callable
from functools import wraps

from cryptography.fernet import Fernet, InvalidToken
from fastapi import Response
from fastapi.responses import RedirectResponse
from init import database

key = Fernet.generate_key()
cipher_suite = Fernet(key)


def login_required(func: Callable[..., Awaitable[Response]]) -> Callable[..., Awaitable[Response]]:
    @wraps(func)
    async def wrapper(*args, **kwargs: dict) -> Response:  # noqa: ANN002
        session: str | None = kwargs.get("session")
        if session is None:
            res = RedirectResponse(url="/authentication", status_code=302)
            res.delete_cookie("session")
            return res

        try:
            user_id = int(cipher_suite.decrypt(session))
        except InvalidToken:
            # не наш токен
            res = RedirectResponse(url="/authentication", status_code=302)
            res.delete_cookie("session")
            return res

        if await database.checker(user_id) is None:
            res = RedirectResponse(url="/authentication", status_code=302)
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
            res = RedirectResponse(url="/", status_code=302)
            res.delete_cookie("session")
            return res

        try:
            user_id = int(cipher_suite.decrypt(session))
        except InvalidToken:
            # не наш токен
            res = RedirectResponse(url="/", status_code=302)
            res.delete_cookie("session")
            return res

        if await database.checker(user_id) is None:
            res = RedirectResponse(url="/", status_code=302)
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

            return RedirectResponse(url="/account", status_code=302)

        return await func(*args, **kwargs)

    return wrapper
