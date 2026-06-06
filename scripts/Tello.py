#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading
import rospy
import cv2
import os
import time
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
        try:
            cv2.imshow("Tello Image", frame)
            cv2.waitKey(1)
        except cv2.error:
            pass

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
        
        self.target_x = -1.3
        self.target_y = -1.8
        self.target_z = 0.7
        self.target_yaw = 90.0
        self.ctrl_thread = threading.Thread(target=self.position_control_demo)
        self.ctrl_thread.daemon = True
        self.ctrl_thread.start()
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
    def position_control_demo(self):
        rate = rospy.Rate(2)  # 控制频率 2 Hz
        print("=== 前往目标位置 Demo 已启动 ===")

        while not rospy.is_shutdown():
            rate.sleep()
            if self.state.position is None or self.state.yaw is None:
                continue

            with self.state.lock:
                px = self.state.position.x
                py = self.state.position.y
                pz = self.state.position.z
                yaw = self.state.yaw

            # ----------------------
            #          YAW角度 控制
            # ----------------------
            yaw_err = self.target_yaw - yaw
            if yaw_err > 180: yaw_err -= 360
            if yaw_err < -180: yaw_err += 360

            if abs(yaw_err) > 10:
                if abs(yaw_err) > 30:
                    step_yaw = 20
                else:
                    step_yaw = 10
                if yaw_err > 0: 
                    self.ccw(step_yaw)
                    print("yaw err: ", yaw_err, ", ccw ", step_yaw)
                    rospy.sleep(0.5)
                else:           
                    self.cw(step_yaw)
                    print("yaw err: ", yaw_err, ", cw ", step_yaw)
                    rospy.sleep(0.5)
                continue

            # ----------------------
            #        位置控制
            # ----------------------
            ex = self.target_x - px
            ey = self.target_y - py
            ez = self.target_z - pz
            # ---- 到达目标 ----
            if abs(ex) < 0.2 and abs(ey) < 0.2 and abs(ez) < 0.2 and abs(yaw_err) < 10:
                print("== 到达目标位置 ==")
                self.stop()
                rate.sleep()
                break

            # X 控制
            if abs(ex) >= 0.2:
                if abs(ex) > 0.8:
                    step_x = 0.5
                else:
                    step_x = 0.2
                if ex > 0: 
                    self.right(int(step_x * 100))
                    print("x err: ", ex, ", right ", step_x * 100)
                else:      
                    self.left(int(step_x * 100))
                    print("x err: ", ex, ", left ", step_x * 100)
                rospy.sleep(0.3)
                continue

            # Y 控制
            if abs(ey) >= 0.2:
                if abs(ey) > 0.8:
                    step_y = 0.5
                else:
                    step_y = 0.2
                if ey > 0: 
                    self.forward(int(step_y * 100))
                    print("y err: ", ey, ", forward ", step_y * 100)
                else:      
                    self.back(int(step_y * 100))
                    print("y err: ", ey, ", back ", step_y * 100)
                rospy.sleep(0.3)
                continue

            # z 控制
            if abs(ez) >= 0.2:
                if abs(ez) > 0.8:
                    step_z = 0.5
                else:
                    step_z = 0.2
                if ez > 0: 
                    self.up(step_z * 100)
                    print("z err: ", ez, ", up ", step_z * 100)
                else:      
                    self.down(step_z * 100)
                    print("z err: ", ez, ", down ", step_z * 100)
                rospy.sleep(0.3)
                continue

            rospy.sleep(0.3)
            # rate.sleep()   #是否可以修改？

    def position_change(self, x, y, z, yaw):
        self.target_x = x
        self.target_y = y
        self.target_z = z
        self.target_yaw = yaw

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

    def detect_fire(self, timeout = 3):
        """检测图像中是否存在红色圆形（火窗标记），返回 True/False"""
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.state.image is None:
                rospy.sleep(0.5)
                if (rospy.Time.now() - start).to_sec() > timeout:
                    return False
                continue

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
            if(rospy.Time.now() - start).to_sec() > timeout:
                return False

    # 白板上 3×3 个目标点的三维世界坐标 (x, y, z)（前方区域)
    # 按行优先排列：第一行从左到右，第二行，第三行
    WHITEBOARD_TARGETS = [
        (-75, 220, 150), (0, 220, 150), (75, 220, 150),
        (-75, 220, 112.5), (0, 220, 112.5), (75, 220, 112.5),
        (-75, 220, 75), (0, 220, 75), (75, 220, 75),
    ]

    # 模板匹配配置：类型 → 模板路径列表
    TEMPLATE_PATHS = {
        "EMPTY":     ["template/EMPTY_1.png", "template/EMPTY_2.png"],
        "LIGHT_ON":  ["template/LIGHT_ON_1.png", "template/LIGHT_ON_2.png"],
        "LIGHT_OFF": ["template/LIGHT_OFF_1.png", "template/LIGHT_OFF_2.png", "template/LIGHT_OFF_3.png", "template/LIGHT_OFF_4.png"],
    }
    TEMPLATE_THRESHOLD = 0.8

    def _match_templates(self, gray):
        """匹配所有模板类型，返回 [(x, y, confidence, type), ...]，已做跨类型 NMS"""
        all_matches = []
        for ttype, paths in self.TEMPLATE_PATHS.items():
            for path in paths:
                tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if tmpl is None:
                    continue
                th, tw = tmpl.shape
                result = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(result >= self.TEMPLATE_THRESHOLD)
                for x, y in zip(xs, ys):
                    all_matches.append((x, y, float(result[y, x]), ttype, tw, th))

        if not all_matches:
            return []

        # 按置信度降序，跨类型 NMS
        all_matches.sort(key=lambda m: -m[2])
        min_dist = min(min(m[4], m[5]) for m in all_matches) // 2
        keep = []
        for m in all_matches:
            if all(np.hypot(m[0] - k[0], m[1] - k[1]) > min_dist for k in keep):
                keep.append(m)
        return keep

    def detect_whiteboard_light(self, timeout=5):
        """
        模板匹配检测白板 3×3 网格，返回 LIGHT_ON 对应的世界坐标 (x, z)。
        检测不到返回 None。
        """
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.state.image is None:
                rospy.sleep(0.5)
                if (rospy.Time.now() - start).to_sec() > timeout:
                    return None
                continue

            frame = self.state.image.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"./image/gray_{timestamp}.jpg", gray)

            matches = self._match_templates(gray)

            # 期望检测到 9 个格子点，允许少量误差
            if len(matches) < 7:
                rospy.sleep(0.5)
                if (rospy.Time.now() - start).to_sec() > timeout:
                    return None
                continue

            # 取置信度最高的若干个点（至少 9 个）
            best = sorted(matches, key=lambda m: -m[2])[:max(len(matches), 9)]

            # 用 x/y 极值确定 3×3 点阵的包围盒
            xs = [m[0] for m in best]
            ys = [m[1] for m in best]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            if x_max - x_min < 1 or y_max - y_min < 1:
                rospy.sleep(0.5)
                continue

            cell_w = (x_max - x_min) / 3.0
            cell_h = (y_max - y_min) / 3.0

            # 将每个点分配到最近的网格单元 (row, col)
            grid_map = {}  # (row, col) → match
            for m in best:
                col = int((m[0] - x_min) / cell_w)
                row = int((m[1] - y_min) / cell_h)
                col = max(0, min(2, col))
                row = max(0, min(2, row))
                key = (row, col)
                if key not in grid_map or m[2] > grid_map[key][2]:
                    grid_map[key] = m

            # 查找 LIGHT_ON 对应的网格位置
            for (row, col), m in grid_map.items():
                if m[3] == "LIGHT_ON":
                    i = row * 3 + col
                    world_x, world_y, world_z = self.WHITEBOARD_TARGETS[i]
                    rospy.loginfo(f"Whiteboard LIGHT_ON at grid({row},{col}) → target ({world_x:.1f}, {world_y:.1f}, {world_z:.1f})")
                    return (world_x, world_y, world_z)

            rospy.logwarn("Whiteboard 3×3 detected but no LIGHT_ON found")
            return None

        return None


    def save_image(self, filename_prefix="capture"):
        """保存当前图像到 U 盘主目录下的 saved_images 文件夹"""
        if self.state.image is None:
            rospy.logwarn("No image to save")
            return None
    
        # 获取 U 盘当前工作目录的根目录（或使用环境变量）
        # 假设 U 盘挂载在 /media 或当前目录就是 U 盘根目录
        usb_root = os.path.expanduser("~")  # 如果当前用户目录就是 U 盘根目录
        # 或者使用 os.getcwd() 获取当前工作目录（通常也是 U 盘内）
        save_dir = os.path.join(usb_root, "saved_images")
    
        # 若不存在则创建
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.jpg"
        filepath = os.path.join(save_dir, filename)
    
        cv2.imwrite(filepath, self.state.image)
        rospy.loginfo(f"Image saved to {filepath}")
        return filepath