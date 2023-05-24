
import Core
from entity.server import Server
from util import config


def initialize():
    server = config.config_load(Server, False)
    Core.server = server
