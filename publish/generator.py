import uuid

from configuration.disk import DiskConfig
from configuration.fastapi import FastApiConfig
from entity.server import Server
from util.config import config_save


def generate():
    try:
        server: Server = Server()
        server.id = str(uuid.uuid4())
        server.sqlUrl = "mysql+pymysql://root:woainiq1@127.0.0.1:3306/tcm"
        server.name = "图像多分类系统"
        server.type = "图像多分类服务器"
        server.description = "图像多分类系统"
        server.diskConfig = DiskConfig()
        server.fastapiConfig = FastApiConfig()
        config_save(server, "../resources")
    except Exception as e:
        print(e)


generate()
