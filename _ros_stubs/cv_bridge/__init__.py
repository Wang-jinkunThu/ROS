"""cv_bridge stub for Windows Pylance."""
from typing import Any
import numpy as np


class CvBridge:
    def __init__(self) -> None: ...
    def imgmsg_to_cv2(self, msg: Any, desired_encoding: str = "passthrough") -> np.ndarray: ...
