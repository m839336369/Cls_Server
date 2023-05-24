import json
import os.path
from typing import Type
import yaml


def config_save(config,path="./resources"):
    if os.path.exists(path) is not True:
        os.mkdir(path)
    with open(f"{path}/{type(config).__name__}.yml", 'w+') as fp:
        return yaml.dump(config, fp,allow_unicode=True)


def config_load(config_cls: Type, default: bool = True,path="./resources"):
    path = f"{path}/{config_cls.__name__}.yml"
    if os.path.exists(path) is not True:
        if not default:
            return dict()
        config = config_cls()
        config_save(config)
        return config
    with open(path, 'r') as fp:
        return yaml.unsafe_load(fp)
