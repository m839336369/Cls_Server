import os

from entity.user import User
from util.mysql import get_session


class UserDao:

    def find(self, id: int) -> User:
        with get_session() as e:
            return e.query(User).filter(User.id == id).first()

    def save(self, user: User):
        with get_session() as e:
            e.add(user)
            e.commit()

    def find_by_username(self, username: str) -> User:
        with get_session() as e:
            return e.query(User).filter(User.username == username).first()
