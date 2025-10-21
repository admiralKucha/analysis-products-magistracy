from controller import customer, guest
from fastapi import APIRouter

api = APIRouter(prefix="")
api.include_router(guest.router)
api.include_router(customer.router)
