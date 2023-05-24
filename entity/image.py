# coding: utf-8
from sqlalchemy import Column, Integer, text, String, LargeBinary, BLOB, Boolean
from sqlalchemy.orm import declarative_base

# 定义实体类
Base = declarative_base()


class Image(Base):
    __tablename__ = 'image'
    __table_args__ = {'comment': '图像表'}
    id = Column(String,primary_key=True, nullable=False, server_default=text("''"), comment='图片ID')
    user_id = Column(String, nullable=False, comment='用户ID')
    image = Column(String, nullable=False, server_default=text("''"), comment='图片')
    cls = Column(String, nullable=False, server_default=text("''"), comment='分类')
    comment = Column(Boolean, nullable=False, server_default=text("''"), comment='分类')
