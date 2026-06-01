"""sensor_msgs.msg stubs for Windows Pylance."""
from typing import Any


class Image:
    header: Any = None
    height: int = 0
    width: int = 0
    encoding: str = ""
    data: bytes = b""
    def __init__(self): ...
