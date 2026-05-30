"""geometry_msgs.msg stub for Windows editing only."""


class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


class Pose:
    position: Point = Point()
    orientation: Quaternion = Quaternion()


class PoseStamped:
    header: object = None
    pose: Pose = Pose()
