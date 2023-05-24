import uuid

from dao.image import ImageDao
from dao.user import UserDao
from entity.image import Image
from entity.user import User


class ImageService:
    def __init__(self):
        self.dao = ImageDao()

    def save(self, user_id, cls, raw_image):
        image = Image()
        image.id = str(uuid.uuid4())
        image_id = image.id
        image.user_id = user_id
        image.cls = cls
        image.image = raw_image
        image.comment = False
        self.dao.save(image)
        return image_id
