from typing import Any

from configuration.disk import DiskConfig
from configuration.fastapi import FastApiConfig
from configuration.uvicorn import UvicornConfig


# 定义服务器数据模型``
class Server:
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.name: str = ""
        self.type: str = ""
        self.id: str = ""
        self.description: str = ""
        self.sqlUrl: str = ""
        self.diskConfig: DiskConfig = DiskConfig()
        self.fastapiConfig: FastApiConfig = FastApiConfig()
        self.uvicornConfig: UvicornConfig = UvicornConfig()