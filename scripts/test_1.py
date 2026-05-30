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
        闭环飞到绝对坐标 (cm)，成功返回 True，超时返回 False。
        """
        rate = rospy.Rate(10)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        rospy.loginfo(f"Going to ({target_x_cm}, {target_y_cm}) cm")

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
            if not self.turn_to(target_deg, tol_deg=5, timeout=5):
                return False

            step = min(dist, 20.0)
            self.forward(int(step))
            rospy.sleep(max(0.5, step / 20.0))
            rate.sleep()

    def turn_to(self, target_yaw_deg, tol_deg=5, timeout=10):
        """
        闭环旋转到指定偏航角（度），成功返回 True。
        """
        rate = rospy.Rate(10)
        deadline = rospy.Time.now() + rospy.Duration(timeout)

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
            if abs(err) < tol_deg:
                self.stop()
                return True
            step = min(int(abs(err)), 20)
            if err > 0:
                self.ccw(step)
            else:
                self.cw(step)
            rospy.sleep(0.5)
            rate.sleep()

    def set_z(self, target_z_cm, tol_cm=5, timeout=15):
        """
        闭环上升到指定高度（cm），成功返回 True。
        """
        rate = rospy.Rate(10)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        rospy.loginfo(f"Setting height to {target_z_cm} cm")

        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                rospy.logwarn(f"set_z timeout at {target_z_cm} cm")
                return False

            with self.state.lock:
                if self.state.position is None:
                    rate.sleep()
                    continue
                pz = self.state.position.z * 100.0

            dz = target_z_cm - pz
            if abs(dz) < tol_cm:
                rospy.loginfo(f"Reached height {target_z_cm} cm")
                self.stop()
                return True

            step = 10
            if abs(dz) > step:
                if dz > 0:
                    self.up(step)
                else:
                    self.down(step)
                rospy.sleep(0.8)
            else:
                if dz > 0:
                    self.up(int(abs(dz)))
                else:
                    self.down(int(abs(dz)))
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

    def det_fire(self):
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


# ============================================================
#  Mission：主任务状态机
# ============================================================
class Mission:
    def __init__(self, drone_name):
        self.uav = TelloControl(drone_name)
        self.state = "TAKEOFF"

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rospy.loginfo(f"Current state: {self.state}")

            try:
                if self.state == "TAKEOFF":
                    self.do_takeoff()
                elif self.state == "BALL":
                    self.ball()
                elif self.state == "WINDOW":
                    self.window()
                elif self.state == "LIGHT":
                    self.light()
                elif self.state == "LANDING":
                    self.landing()
                elif self.state == "FINISH":
                    rospy.loginfo("Mission finished")
                    break
                else:
                    rospy.logerr(f"Unknown state: {self.state}")
                    break
            except Exception as e:
                rospy.logerr(f"Exception in state '{self.state}': {e}")
                rospy.loginfo("Emergency landing...")
                self.state = "FINISH"
                self.uav.land()

            rate.sleep()

    # ========== 各状态实现 ==========
    def do_takeoff(self):
        rospy.loginfo("Taking off...")
        self.uav.takeoff()
        rospy.sleep(3)

        drone_name = self.uav.state.name
        if drone_name:
            service_name = f"/{drone_name}/set_downvision"
        else:
            service_name = "/set_downvision"
        rospy.loginfo(f"Waiting for service: {service_name}")
        rospy.wait_for_service(service_name)
        try:
            set_down = rospy.ServiceProxy(service_name, SetBool)
            resp = set_down(True)
            if resp.success:
                rospy.loginfo("Downvision enabled successfully")
            else:
                rospy.logwarn(f"Downvision enable failed: {resp.message}")
        except Exception as e:
            rospy.logerr(f"Failed to call downvision service: {e}")

        self.state = "BALL"

    def ball(self):
        self.uav.goto_xy(150, 125)
        self.uav.set_z(70)
        self.uav.turn_to(0)

        ball1_color = self.uav.wait_for_ball(timeout=10)
        if ball1_color is None:
            rospy.logwarn("Rotating ball not detected, set to unknown")
            ball1_color = '?'
        rospy.loginfo(f"Rotating ball color: {ball1_color}")

        self.uav.goto_xy(150, 50)
        self.uav.goto_xy(200, 50)
        self.uav.goto_xy(310, 125)

        ball2_color = self.uav.wait_for_ball(timeout=5)
        if ball2_color is None:
            rospy.logwarn("Fixed ball not detected, set to unknown")
            ball2_color = '?'
        rospy.loginfo(f"Fixed ball color: {ball2_color}")

        ball_result = f"{ball1_color}{ball2_color}"
        rospy.loginfo(f"Ball detection result: {ball_result}")
        self.uav.judge_pub.publish(ball_result)
        print(f"[JUDGE] Ball result sent: {ball_result}")

        rospy.sleep(1)
        self.state = "WINDOW"

    def window(self):
        detect_points = [(250, 235), (150, 235), (350, 235)]
        fire_found = False
        selected_x = None
        self.uav.set_z(175)

        for (wx, wy) in detect_points:
            rospy.loginfo(f"Checking at ({wx}, {wy})")
            if not self.uav.goto_xy(wx, wy):
                continue
            self.uav.turn_to(90)
            rospy.sleep(0.5)

            if self.uav.det_fire():
                rospy.loginfo(f"Fire window detected at ({wx}, {wy})")
                fire_found = True
                selected_x = wx
                break

        if not fire_found:
            rospy.logwarn("No fire window detected, using first point as default")
            selected_x = detect_points[0][0]

        self.uav.set_z(148)
        self.uav.goto_xy(selected_x, 235)

        rospy.sleep(2)
        self.state = "LIGHT"

    def light(self):
        for i in range(3):
            self.uav.goto_xy(250, 400)
            self.uav.set_z(125)
            self.uav.turn_to(90)

            corners = None
            while corners is None and not rospy.is_shutdown():
                corners = self.uav.detect_whiteboard_corners()
                rospy.sleep(0.5)
            if corners is None:
                rospy.logerr("Cannot detect whiteboard")
                self.state = "LANDING"
                return

            pixel = None
            timeout = rospy.Time.now() + rospy.Duration(30)
            while pixel is None and not rospy.is_shutdown():
                pixel = self.uav.detect_red_light()
                if rospy.Time.now() > timeout:
                    break
                rospy.sleep(0.5)
            if pixel is None:
                rospy.logwarn("No red light detected, skip")
                continue

            x_world, z_world = self.uav.pixel_to_world(pixel[0], pixel[1], corners)
            rospy.loginfo(f"Go to x={x_world}, z={z_world}")
            self.uav.goto_xy(x_world, 500)
            self.uav.set_z(z_world)

            target_msg = f"target{i + 1}"
            self.uav.judge_pub.publish(target_msg)
            print(f"[JUDGE] Target sent: {target_msg}")
            rospy.sleep(1)

        rospy.sleep(2)
        self.state = "LANDING"

    def landing(self):
        rospy.loginfo("Landing...")
        self.uav.goto_xy(470, 470)
        self.uav.set_z(30)
        self.uav.land()
        rospy.sleep(3)
        self.state = "FINISH"


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    rospy.init_node("tello_control_node", anonymous=False)
    name = rospy.get_param('~name', "")
    mission = Mission(name)
    mission.run()
