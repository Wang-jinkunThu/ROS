#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from std_srvs.srv import SetBool


# ============================================================
#  TelloState：图像 + 位姿的状态类
# ============================================================
class TelloState:
    def __init__(self, name):
        self.bridge = CvBridge()

        self.image = None
        self.position = None
        self.orientation = None
        self.yaw = None
        self.name = name
        self.lock = threading.Lock()

        rospy.Subscriber(f"{name}/image_raw", Image, self.image_callback)
        rospy.Subscriber(f"{name}/pose", PoseStamped, self.pose_callback)

    def quat_to_yaw(self, q):
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        return math.degrees(math.atan2(siny_cosp, cosy_cosp))

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.image = frame

    def pose_callback(self, msg: PoseStamped):
        yaw_deg = self.quat_to_yaw(msg.pose.orientation)
        with self.lock:
            self.position = msg.pose.position
            self.orientation = msg.pose.orientation
            self.yaw = yaw_deg


# ============================================================
#  TelloControl：控制 + 裁判机通信
# ============================================================
class TelloControl:
    def __init__(self, name):
        rospy.loginfo(f"Tello Name: {name}")
        self.state = TelloState(name)

        self.pub = rospy.Publisher(f"{name}/sdk_cmd", String, queue_size=1)
        self.judge_pub = rospy.Publisher(f"{name}/judge", String, queue_size=1)

    # ========== 基本控制指令 ==========
    def send(self, cmd):
        self.pub.publish(cmd)

    def forward(self, cm):  self.send(f"forward {cm}")
    def back(self, cm):     self.send(f"back {cm}")
    def left(self, cm):     self.send(f"left {cm}")
    def right(self, cm):    self.send(f"right {cm}")
    def up(self, cm):       self.send(f"up {cm}")
    def down(self, cm):     self.send(f"down {cm}")
    def cw(self, deg):      self.send(f"cw {deg}")
    def ccw(self, deg):     self.send(f"ccw {deg}")
    def stop(self):         self.send("stop")
    def takeoff(self):      self.send("takeoff")
    def land(self):         self.send("land")

    # ========== 闭环控制方法 ==========
    def goto_xy(self, target_x_cm, target_y_cm, tol_cm=5, timeout=30):
        """
        闭环飞到绝对坐标 (cm),成功返回 True,超时返回 False.
        """
        rate = rospy.Rate(10)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        rospy.loginfo(f"Going to ({target_x_cm}, {target_y_cm}) cm")
        turn_failures = 0

        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                rospy.logwarn(f"goto_xy timeout at ({target_x_cm}, {target_y_cm})")
                return False

            with self.state.lock:
                if self.state.position is None:
                    rate.sleep()
                    continue
                px = self.state.position.x * 100.0
                py = self.state.position.y * 100.0

            dx = target_x_cm - px
            dy = target_y_cm - py
            dist = math.hypot(dx, dy)
            if dist < tol_cm:
                rospy.loginfo(f"Reached ({target_x_cm}, {target_y_cm})")
                self.stop()
                return True

            target_rad = math.atan2(dy, dx)
            target_deg = math.degrees(target_rad)
            if not self.turn_to(target_deg, tol_deg=5, timeout=10):
                turn_failures += 1
                if turn_failures >= 3:
                    rospy.logerr(f"turn_to failed {turn_failures} times, giving up")
                    return False
                rospy.logwarn(f"turn_to failed ({turn_failures}/3), retrying...")
                rospy.sleep(1)
                continue
            turn_failures = 0

            step = min(dist, 20.0)
            self.forward(int(step))
            rospy.sleep(max(0.3, step / 30.0))

    def turn_to(self, target_yaw_deg, tol_deg=5, timeout=10):
        """
        闭环旋转到指定偏航角（度），成功返回 True。
        """
        rate = rospy.Rate(10)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        prev_yaw = None
        stable_count = 0

        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                rospy.logwarn(f"turn_to timeout at {target_yaw_deg} deg")
                return False

            with self.state.lock:
                if self.state.yaw is None:
                    rate.sleep()
                    continue
                yaw = self.state.yaw
            err = target_yaw_deg - yaw
            if err > 180:
                err -= 360
            if err < -180:
                err += 360

            # 稳定性检查：连续 3 次在容差内才算收敛
            if abs(err) < tol_deg:
                stable_count += 1
                if stable_count >= 3:
                    self.stop()
                    return True
            else:
                stable_count = 0

            # 检测异常跳变：yaw 变化远超命令量，等待下一次读数
            if prev_yaw is not None:
                yaw_delta = abs(yaw - prev_yaw)
                if yaw_delta > 25:
                    rospy.loginfo_throttle(1.0, f"turn_to: suspicious yaw jump {prev_yaw:.1f}->{yaw:.1f}, waiting")
                    prev_yaw = yaw
                    rospy.sleep(0.3)
                    continue

            step = min(int(abs(err) * 0.6), 10)
            step = max(step, 3)
            rospy.loginfo_throttle(0.5, f"turn_to: yaw={yaw:.1f}  target={target_yaw_deg:.1f}  err={err:.1f}  step={step}  dir={'ccw' if err > 0 else 'cw'}")
            if err > 0:
                self.ccw(step)
            else:
                self.cw(step)
            prev_yaw = yaw
            rospy.sleep(0.5)

    def move_z(self, delta_cm, tol_cm=5, timeout=15):
        """z 方向上移动相对高度，delta_cm > 0 上升，< 0 下降，成功返回 True。"""
        rate = rospy.Rate(10)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        stable_count = 0

        # 尝试获取当前高度，拿不到则先发一次移动指令让无人机动起来
        init_z = None
        with self.state.lock:
            if self.state.position is not None:
                init_z = self.state.position.z * 100.0

        if init_z is None:
            # 盲发第一次指令，触发运动使 pose 数据出现
            if delta_cm > 0:
                self.up(min(abs(delta_cm), 20))
            else:
                self.down(min(abs(delta_cm), 20))
            rospy.loginfo("move_z: sent initial command, waiting for pose...")
            for _ in range(30):
                rospy.sleep(0.5)
                with self.state.lock:
                    if self.state.position is not None:
                        init_z = self.state.position.z * 100.0
                        break

        if init_z is None:
            rospy.logerr("move_z: cannot get initial height")
            return False

        target_z = init_z + delta_cm
        rospy.loginfo(f"move_z: delta={delta_cm} cm  (from {init_z:.1f} to {target_z:.1f} cm)")

        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                rospy.logwarn(f"move_z timeout: delta={delta_cm} cm  target={target_z:.1f}")
                return False

            with self.state.lock:
                if self.state.position is None:
                    rate.sleep()
                    continue
                pz = self.state.position.z * 100.0

            dz = target_z - pz
            if abs(dz) < tol_cm:
                stable_count += 1
                if stable_count >= 3:
                    rospy.loginfo(f"Reached height {pz:.1f} cm")
                    self.stop()
                    return True
            else:
                stable_count = 0

            step = max(min(int(abs(dz) * 0.5), 50), 20)
            if dz > 0:
                self.up(step)
            else:
                self.down(step)
            rospy.sleep(0.5)

    def _map_world_axis(self, delta, yaw):
        """根据 yaw 选择机体指令来逼近世界坐标轴移动。
        返回 (cmd, step)，cmd 为 'forward'/'back'/'left'/'right'。"""
        yaw = yaw % 360
        if yaw > 180:
            yaw -= 360

        if -45 <= yaw <= 45:           # 机头朝 +X 附近
            return ('forward' if delta > 0 else 'back')
        elif 45 < yaw <= 135:          # 机头朝 +Y 附近
            return ('right' if delta > 0 else 'left')
        elif yaw > 135 or yaw < -135:  # 机头朝 -X 附近
            return ('back' if delta > 0 else 'forward')
        else:                          # -135 <= yaw < -45，机头朝 -Y 附近
            return ('left' if delta > 0 else 'right')

    def _yaw_map_cmd(self, cmd, step):
        if cmd == 'forward':   self.forward(step)
        elif cmd == 'back':    self.back(step)
        elif cmd == 'left':    self.left(step)
        elif cmd == 'right':   self.right(step)

    def move_x(self, delta_cm, tol_cm=5, timeout=15):
        """x 方向上水平移动相对距离（世界坐标系），delta_cm > 0 向 +x 方向，成功返回 True。"""
        rate = rospy.Rate(10)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        stable_count = 0

        init_x = None
        with self.state.lock:
            if self.state.position is not None:
                init_x = self.state.position.x * 100.0

        if init_x is None:
            if delta_cm > 0:
                self.right(min(abs(delta_cm), 30))
            else:
                self.left(min(abs(delta_cm), 30))
            rospy.loginfo("move_x: sent initial command, waiting for pose...")
            for _ in range(30):
                rospy.sleep(0.5)
                with self.state.lock:
                    if self.state.position is not None:
                        init_x = self.state.position.x * 100.0
                        break

        if init_x is None:
            rospy.logerr("move_x: cannot get initial position")
            return False

        target_x = init_x + delta_cm
        rospy.loginfo(f"move_x: delta={delta_cm} cm  (from x={init_x:.1f} to x={target_x:.1f} cm)")

        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                rospy.logwarn(f"move_x timeout: delta={delta_cm} cm  target_x={target_x:.1f}")
                return False

            with self.state.lock:
                if self.state.position is None or self.state.yaw is None:
                    rate.sleep()
                    continue
                px = self.state.position.x * 100.0
                yaw = self.state.yaw

            dx = target_x - px
            if abs(dx) < tol_cm:
                stable_count += 1
                if stable_count >= 3:
                    rospy.loginfo(f"Reached x={px:.1f} cm")
                    self.stop()
                    return True
            else:
                stable_count = 0

            step = max(min(int(abs(dx) * 0.5), 50), 20)
            cmd = self._map_world_axis(dx, yaw)
            self._yaw_map_cmd(cmd, step)
            rospy.sleep(0.5)

    def move_y(self, delta_cm, tol_cm=5, timeout=15):
        """y 方向上水平移动相对距离（世界坐标系），delta_cm > 0 向 +y 方向，成功返回 True。"""
        rate = rospy.Rate(10)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        stable_count = 0

        init_y = None
        with self.state.lock:
            if self.state.position is not None:
                init_y = self.state.position.y * 100.0

        if init_y is None:
            if delta_cm > 0:
                self.forward(min(abs(delta_cm), 30))
            else:
                self.back(min(abs(delta_cm), 30))
            rospy.loginfo("move_y: sent initial command, waiting for pose...")
            for _ in range(30):
                rospy.sleep(0.5)
                with self.state.lock:
                    if self.state.position is not None:
                        init_y = self.state.position.y * 100.0
                        break

        if init_y is None:
            rospy.logerr("move_y: cannot get initial position")
            return False

        target_y = init_y + delta_cm
        rospy.loginfo(f"move_y: delta={delta_cm} cm  (from y={init_y:.1f} to y={target_y:.1f} cm)")

        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                rospy.logwarn(f"move_y timeout: delta={delta_cm} cm  target_y={target_y:.1f}")
                return False

            with self.state.lock:
                if self.state.position is None or self.state.yaw is None:
                    rate.sleep()
                    continue
                py = self.state.position.y * 100.0
                yaw = self.state.yaw

            dy = target_y - py
            if abs(dy) < tol_cm:
                stable_count += 1
                if stable_count >= 3:
                    rospy.loginfo(f"Reached y={py:.1f} cm")
                    self.stop()
                    return True
            else:
                stable_count = 0

            step = max(min(int(abs(dy) * 0.5), 50), 20)
            # y 轴映射：在 x 轴基础上偏移 90°
            yaw_shifted = yaw - 90
            cmd = self._map_world_axis(dy, yaw_shifted)
            self._yaw_map_cmd(cmd, step)
            rospy.sleep(0.5)

    def set_z(self, target_cm, tol_cm=20, timeout=15):
        """飞到指定绝对高度 target_cm（cm），成功返回 True。"""
        rate = rospy.Rate(10)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        stable_count = 0

        init_z = None
        for _ in range(30):
            with self.state.lock:
                if self.state.position is not None:
                    init_z = self.state.position.z * 100.0
                    break
            rospy.sleep(0.2)

        if init_z is None:
            rospy.logerr("set_z: cannot get current height")
            return False

        rospy.loginfo(f"set_z: target={target_cm:.1f} cm  (current={init_z:.1f} cm)")

        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                rospy.logwarn(f"set_z timeout: target={target_cm:.1f} cm")
                return False

            with self.state.lock:
                if self.state.position is None:
                    rate.sleep()
                    continue
                pz = self.state.position.z * 100.0

            dz = target_cm - pz
            if abs(dz) < tol_cm:
                stable_count += 1
                if stable_count >= 3:
                    rospy.loginfo(f"set_z done: {pz:.1f} cm")
                    self.stop()
                    return True
            else:
                stable_count = 0

            step = max(min(int(abs(dz) * 0.5), 50), 20)
            if dz > 0:
                self.up(step)
            else:
                self.down(step)
            rospy.sleep(0.5)

    # ========== 视觉检测方法 ==========
    def detect_ball_color(self):
        """检测图像中球的颜色，返回 'g' 或 'r'，检测不到返回 None"""
        if self.state.image is None:
            return None
        frame = self.state.image.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 100, 80])
        upper_green = np.array([45, 255, 255])
        lower_orange = np.array([5, 80, 70])
        upper_orange = np.array([30, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel)
        green_area = cv2.countNonZero(mask_green)
        orange_area = cv2.countNonZero(mask_orange)
        if max(green_area, orange_area) < 200:
            return None
        return 'g' if green_area > orange_area else 'r'

    def wait_for_ball(self, timeout=10):
        """等待并检测球的颜色，返回颜色字符，超时返回 None"""
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            color = self.detect_ball_color()
            if color is not None:
                return color
            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.logwarn("Ball detection timeout")
                return None
            rospy.sleep(0.5)

    def detect_fire(self):
        """检测图像中是否存在红色圆形（火窗标记），返回 True/False"""
        if self.state.image is None:
            rospy.logwarn("No image received yet")
            return False

        frame = self.state.image.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            (x_center, y_center), radius = cv2.minEnclosingCircle(cnt)
            if radius == 0:
                continue
            circularity = area / (math.pi * radius * radius)
            if circularity > 0.7:
                rospy.loginfo("Fire window detected!")
                return True
        return False

    def detect_whiteboard_corners(self):
        """
        检测白板区域，返回四个角点的像素坐标 [左上,右上,右下,左下]
        世界坐标：左上(150,164) 右上(350,164) 右下(350,75) 左下(150,75)
        """
        if self.state.image is None:
            return None
        frame = self.state.image.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        max_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)
        if len(approx) != 4:
            hull = cv2.convexHull(max_contour)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
            if len(approx) != 4:
                return None
        pts = approx.reshape(4, 2)
        pts_sorted = sorted(pts, key=lambda p: (p[1], p[0]))
        top_pts = sorted(pts_sorted[:2], key=lambda p: p[0])
        bottom_pts = sorted(pts_sorted[2:], key=lambda p: p[0])
        corners = top_pts + bottom_pts
        return np.array(corners, dtype='float32')

    def pixel_to_world(self, u, v, corners):
        """
        通过白板角点计算单应性矩阵，将像素坐标 (u,v) 映射到世界坐标 (x,z)
        """
        world_pts = np.array([
            [150, 164],
            [350, 164],
            [350, 75],
            [150, 75]
        ], dtype='float32')

        H, _ = cv2.findHomography(corners, world_pts)
        pixel = np.array([u, v, 1], dtype='float32').reshape(3, 1)
        world = H @ pixel
        world = world / world[2]
        x = world[0, 0]
        z = world[1, 0]
        x = max(150, min(350, x))
        z = max(75, min(164, z))
        return x, z

    def detect_red_light(self):
        """检测图像中的红色光点（红灯），返回像素坐标 (u, v) 或 None"""
        if self.state.image is None:
            return None
        frame = self.state.image.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        best_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                best_cnt = cnt
        if best_cnt is not None:
            M = cv2.moments(best_cnt)
            if M["m00"] != 0:
                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])
                return (u, v)
        return None
