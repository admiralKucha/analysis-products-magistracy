import logging
import pickle
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

from db import customer_db, guest_db, main_db, product_db

# фронтенд
templates = Jinja2Templates(directory="templates")

# база данных
database = main_db.PostgresDB()
database_guest = guest_db.PostgresDBGuest()
database_customer = customer_db.PostgresDBCustomer()
database_products = product_db.PostgresDBProduct()
apriori2_rules = None
k_means_rules = None
top1000_products = None

logging.basicConfig(
    level=logging.INFO,  # Уровень логирования
    format="%(levelname)s - %(asctime)s - %(message)s",  # Формат сообщения
    handlers=[
        logging.FileHandler("../backups/app.log"),  # Запись логов в файл
    ],
)


@asynccontextmanager
async def startup(app: FastAPI) -> AsyncGenerator[None, None]:
    global apriori2_rules, k_means_rules, top1000_products

    # Инициализация пула соединений с базой данных
    await database.create_pool()
    await database_guest.create_pool()
    await database_customer.create_pool()
    await database_products.create_pool()
    with open("/backend/rec_systems/results/apriori2_dict.pkl", "rb") as f:
        apriori2_rules = pickle.load(f)

    with open("/backend/rec_systems/results/k_means2.pkl", "rb") as f:
        k_means_rules = pickle.load(f)

    with open("/backend/rec_systems/results/chastota.pickle", "rb") as f:
        top1000_products = pickle.load(f)
        top1000_products = sorted(top1000_products.keys(), key=lambda x: int(top1000_products[x]), reverse=True)[:1000]

    app.mount("/static", StaticFiles(directory="static"), name="static")
    yield

    # Отключаемся от всех соединений
    await database_guest.delete_pool()
    await database.delete_pool()
    await database_customer.delete_pool()
    await database_products.delete_pool()
    apriori2_rules = None
    k_means_rules = None
    top1000_products = None
