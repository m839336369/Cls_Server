import os

from entity.image import Image
from entity.user import User
from util.mysql import get_session


class ImageDao:

    def find_by_user_id(self, user_id: str) -> list:
        with get_session() as e:
            return e.query(Image).filter(Image.user_id == user_id).all()

    def find_by_id(self, id: str) -> Image:
        with get_session() as e:
            return e.query(Image).filter(Image.id == id).first()

    def save(self, image: Image):
        with get_session() as e:
            e.add(image)
            e.commit()

    def updateComment(self, image: Image):
        with get_session() as e:
            e.query(Image).filter(Image.id == image.id).update({
                'comment': image.comment
            })
            e.commit()

    def delete_by_image_id(self, image_id):
        with get_session() as e:
            e.query(Image).filter(Image.id == image_id).delete()
            e.commit()
