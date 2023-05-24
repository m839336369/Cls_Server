# 创建数据库引擎和会话
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import Core

engine = create_engine(Core.server.sqlUrl, echo=True)
Session = sessionmaker(bind=engine)

def get_session():
    return Session()
