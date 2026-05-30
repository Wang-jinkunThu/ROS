"""cv_bridge stub for Windows editing only."""
import numpy as np


class CvBridge:
    def imgmsg_to_cv2(self, msg, desired_encoding: str = "bgr8") -> np.ndarray: ...
    def cv2_to_imgmsg(self, img, encoding: str = "bgr8"): ...
