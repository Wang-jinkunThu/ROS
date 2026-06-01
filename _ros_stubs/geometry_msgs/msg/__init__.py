"""geometry_msgs.msg stubs for Windows Pylance."""
from typing import Any


class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Pose:
    position: Point
    orientation: Quaternion
    def __init__(self): ...


class Header:
    seq: int = 0
    stamp: Any = None
    frame_id: str = ""
    def __init__(self): ...


class PoseStamped:
    header: Header
    pose: Pose
    def __init__(self): ...


class Twist:
    linear: Vector3
    angular: Vector3
    def __init__(self): ...
