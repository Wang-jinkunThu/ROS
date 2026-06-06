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

    # 白板上 3×3 个目标点的世界坐标 (x, z)，y=220 恒定
    # 左上为原点，x 右为正，z 下为正（世界坐标系中 z↓ 即高度↓）
    WHITEBOARD_TARGETS = [
        (-75, 150), (0, 150), (75, 150),
        (-75, 112.5), (0, 112.5), (75, 112.5),
        (-75, 75), (0, 75), (75, 75),
    ]

    def detect_whiteboard_corners(self, timeout=5):
        """
        检测白板区域，返回四个角点的像素坐标 [左上,右上,右下,左下]
        超时时间 timeout 秒，超时返回 None
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
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                rospy.sleep(0.5)
                if (rospy.Time.now() - start).to_sec() > timeout:
                    return None
                continue

            max_contour = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)
            if len(approx) != 4:
                hull = cv2.convexHull(max_contour)
                approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
                if len(approx) != 4:
                    rospy.sleep(0.5)
                    if (rospy.Time.now() - start).to_sec() > timeout:
                        return None
                    continue

            pts = approx.reshape(4, 2)
            pts_sorted = sorted(pts, key=lambda p: (p[1], p[0]))
            top_pts = sorted(pts_sorted[:2], key=lambda p: p[0])          # [TL, TR]
            bottom_pts = sorted(pts_sorted[2:], key=lambda p: -p[0])      # [BR, BL]
            corners = top_pts + bottom_pts                                 # [TL, TR, BR, BL]
            return np.array(corners, dtype='float32')
        


        return None

    def pixel_to_world(self, u, v, corners):
        """将像素坐标 (u,v) 映射到世界坐标 (x,z)（白板平面 y=220），并吸附到最近的目标点。"""
        world_pts = np.array([
            [-75, 150],  # 左上
            [75, 150],   # 右上
            [75, 75],    # 右下
            [-75, 75]    # 左下
        ], dtype='float32')

        H, _ = cv2.findHomography(corners, world_pts)
        pixel = np.array([u, v, 1], dtype='float32').reshape(3, 1)
        world = H @ pixel
        world = world / world[2]
        x = world[0, 0]
        z = world[1, 0]
        x = max(-75, min(75, x))
        z = max(75, min(150, z))

        # 吸附到最近的 3×3 网格点
        best = min(self.WHITEBOARD_TARGETS, key=lambda p: (p[0] - x)**2 + (p[1] - z)**2)
        return best[0], best[1]

    def detect_red_light(self, board_corners=None, min_area=10, min_circularity=0.7, timeout=5):
        """
        检测图像中的红色光点，返回像素坐标 (u, v) 或 None
        超时时间 timeout 秒，超时返回 None
        """
        
        start = rospy.Time.now()
        
        while not rospy.is_shutdown():
            if self.state.image is None:
                rospy.sleep(0.5)
                if (rospy.Time.now() - start).to_sec() > timeout:
                    return None
                continue

            frame = self.state.image.copy()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 10, 40])
            upper_red1 = np.array([20, 255, 255])
            lower_red2 = np.array([150, 10, 40])
            upper_red2 = np.array([180, 255, 255])
            mask = cv2.bitwise_or(
                cv2.inRange(hsv, lower_red1, upper_red1),
                cv2.inRange(hsv, lower_red2, upper_red2)
            )

            if cv2.countNonZero(mask) < min_area:
                rospy.sleep(0.5)
                if (rospy.Time.now() - start).to_sec() > timeout:
                    return None
                continue

            kernel = np.ones((5, 5), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            try:
                cv2.imshow("red_mask", mask)
                cv2.waitKey(1)
            except cv2.error:
                pass

            best_score = -1
            best_uv = None
            max_area = 500
            hull = None
            if board_corners is not None:
                hull = cv2.convexHull(board_corners.astype(np.float32))
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue
                peri = cv2.arcLength(cnt, True)
                if peri == 0:
                    continue
                circularity = 4 * math.pi * area / (peri * peri)
                if circularity < min_circularity:
                    continue

                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])

                if hull is not None:
                    inside_dist = cv2.pointPolygonTest(hull, (float(u), float(v)), True)
                    if inside_dist < 10:
                        continue

                # 评分：圆形度权重高，面积辅助
                score = circularity * 1000 + area
                if score > best_score:
                    best_score = score
                    best_uv = (u, v)

            if best_uv is not None:
                return best_uv

            rospy.sleep(0.5)
            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.loginfo("detect light timeout!")
                break

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