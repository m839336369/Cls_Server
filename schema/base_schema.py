import time
from typing import List, Dict, Any

from pydantic import BaseModel

from util.json_encoders import JSONEncoders


class ResponseSchema(BaseModel):
    """
    基础返回Schema
    """
    code: int = 1  # 返回编号
    timestamp = time.time()
    """
    基础Schema
    """

    def __init__(self, **data: Any):
        super().__init__(**data)

    class Config:
        json_encoders = JSONEncoders.json_encoders  # 使用自定义json转换


class SuccessSchema(ResponseSchema):
    """
    基础返回Schema
    """
    code: int = 200  # 返回编号

    class Config:
        json_encoders = JSONEncoders.json_encoders  # 使用自定义json转换


class SuccessResponseSchema(ResponseSchema):
    """
    基础返回Schema
    """
    code: int = 200  # 返回编号
    data: object  # 返回编号

    def __init__(self, data: object, **param: Any):
        super().__init__(**param)
        self.data = data  # 返回编号

    class Config:
        json_encoders = JSONEncoders.json_encoders  # 使用自定义json转换


class ErrorSchema(ResponseSchema):
    """
    基础返回Schema
    """
    code: int = 500  # 返回编号

    class Config:
        json_encoders = JSONEncoders.json_encoders  # 使用自定义json转换


class ErrorResponseSchema(ResponseSchema):
    """
    基础返回Schema
    """
    code: int = 500  # 返回编号
    data: object  # 返回编号

    def __init__(self, data: object, **args: Any):
        super().__init__(**args)
        self.data = data

    class Config:
        json_encoders = JSONEncoders.json_encoders  # 使用自定义json转换
