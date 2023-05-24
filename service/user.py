import uuid

from dao.user import UserDao
from entity.user import User


class UserService:
    def __init__(self):
        self.dao = UserDao()

    def register(self, username, password):
        user = User()
        user.username = username
        user.password = password
        user.id = uuid.uuid4()
        self.dao.save(user)

    def login(self, username, password):
        checkUser = self.dao.find_by_username(username)
        if checkUser is not None and checkUser.password == password:
            return checkUser.id
        else:
            return None
