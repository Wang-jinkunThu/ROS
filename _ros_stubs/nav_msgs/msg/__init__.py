"""nav_msgs.msg stubs for Windows Pylance."""
from typing import Any
from geometry_msgs.msg import Pose, Twist


class PoseWithCovariance:
    pose: Pose
    def __init__(self): ...


class TwistWithCovariance:
    twist: Twist
    def __init__(self): ...


class Header:
    seq: int = 0
    stamp: Any = None
    frame_id: str = ""
    def __init__(self): ...


class Odometry:
    header: Header
    pose: PoseWithCovariance
    twist: TwistWithCovariance
    def __init__(self): ...
