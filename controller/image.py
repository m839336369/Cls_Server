import base64
import io

from PIL import Image

from fastapi import UploadFile, APIRouter
from fastapi.params import File, Form, Query, Param

from predict_api import predict_image
from schema.base_schema import ErrorResponseSchema, SuccessResponseSchema, SuccessSchema
from service.image import ImageService
from service.user import UserService

router = APIRouter()
user_service = UserService()
image_service = ImageService()


@router.post("/cls")
async def cls(user_id: str = Form(...), file: str = Form(...)):
    if user_service.dao.find(user_id) is None:
        return ErrorResponseSchema("用户名不存在")
    data = file[file.index(',') + 1:]  # 去掉图片Base64的标记前缀
    result = predict_image(image=Image.open(io.BytesIO(base64.b64decode(data))))
    image_id = image_service.save(user_id, result, file)
    image = image_service.dao.find_by_id(image_id)
    return SuccessResponseSchema(image)


@router.get("/images")
async def getImages(user_id: str = Param(...)):
    if user_service.dao.find(user_id) is None:
        return ErrorResponseSchema("用户名不存在")
    images = image_service.dao.find_by_user_id(user_id)
    return SuccessResponseSchema(images)


@router.post("/comment")
async def comment(image_id: str = Form(...), image_boolean: bool = Form(...)):
    image = image_service.dao.find_by_id(image_id)
    if image is None:
        return ErrorResponseSchema("照片不存在")
    image.comment = image_boolean
    image_service.dao.updateComment(image)
    return SuccessSchema()


@router.put("/delete")
async def delete(image_id: str = Form(...)):
    image_service.dao.delete_by_image_id(image_id)
    return SuccessSchema()
