from PIL import Image

from fastapi import UploadFile, APIRouter
from fastapi.params import File, Form
from predict_api import predict_image
from schema.base_schema import SuccessResponseSchema, ErrorSchema, SuccessSchema
from service.user import UserService

router = APIRouter()

service = UserService()

@router.post("/login")
async def login(username: str = Form(), password: str = Form()):
    id = service.login(username,password)
    if id is not None:
        return SuccessResponseSchema(id)
    else:
        return ErrorSchema()


@router.post("/register")
async def register(username: str = Form(), password: str = Form()):
    result = service.register(username,password)
    if result == True:
        return SuccessSchema()
    else:
        return ErrorSchema()
