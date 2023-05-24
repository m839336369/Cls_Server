# coding: utf-8
from sqlalchemy import Column, Integer, text, String
from sqlalchemy.orm import declarative_base

# 定义实体类
Base = declarative_base()


class User(Base):
    __tablename__ = 'user'
    __table_args__ = {'comment': '用户表'}
    id = Column(Integer, nullable=False, primary_key=True, index=True, comment='用户ID')
    username = Column(String, nullable=False, server_default=text("''"), comment='账号')
    password = Column(String, nullable=False, server_default=text("''"), comment='密码')
