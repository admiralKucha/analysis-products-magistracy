import json

from controller.api import api
from fastapi import FastAPI, Response
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from init import startup
from starlette.requests import Request

app = FastAPI(lifespan=startup)

app.include_router(api)


@app.exception_handler(RequestValidationError)
async def http_exception_handler(request: Request, exc: RequestValidationError) -> Response:  # noqa: RUF029 # можешь попробовать убрать
    errors = exc.errors()
    res = {"status": "error", "message": "Ошибка валидации", "detail": []}
    for error in errors:
        # Добавляет каждую ошибку в список
        res["detail"].append(
            {
                "loc": error.get("loc", ["", "Неизвестная ошибка"])[1:],
                "type": error.get("type", "Неизвестная ошибка"),
            }
        )

    return Response(
        content=json.dumps(res, ensure_ascii=False),
        status_code=422,
        media_type="application/json",
    )


def custom_openapi() -> dict:
    if not app.openapi_schema:
        app.openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            terms_of_service=app.terms_of_service,
            contact=app.contact,
            license_info=app.license_info,
            routes=app.routes,
            tags=app.openapi_tags,
            servers=app.servers,
        )
        for method_item in app.openapi_schema.get("paths").values():
            for param in method_item.values():
                responses = param.get("responses")
                # remove 422 response, also can remove other status code
                if "422" in responses and responses.get("422").get("content", None) is None:
                    del responses["422"]
    return app.openapi_schema


app.openapi = custom_openapi
