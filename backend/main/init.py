import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

from db import guest_db, main_db

# фронтенд
templates = Jinja2Templates(directory="templates")

# база данных
database = main_db.PostgresDB()
database_guest = guest_db.PostgresDBGuest()

logging.basicConfig(
    level=logging.INFO,  # Уровень логирования
    format="%(levelname)s - %(asctime)s - %(message)s",  # Формат сообщения
    handlers=[
        logging.FileHandler("../backups/app.log"),  # Запись логов в файл
    ],
)


@asynccontextmanager
async def startup(app: FastAPI) -> AsyncGenerator[None, None]:
    # Инициализация пула соединений с базой данных
    await database.create_pool()
    await database_guest.create_pool()
    app.mount("/static", StaticFiles(directory="static"), name="static")
    yield

    # Отключаемся от всех соединений
    await database_guest.delete_pool()
    await database.delete_pool()
