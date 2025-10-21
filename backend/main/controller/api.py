from controller import guest
from fastapi import APIRouter

api = APIRouter(prefix="")
api.include_router(guest.router)
