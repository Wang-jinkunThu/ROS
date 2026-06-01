#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

# ======================== 工具函数 ========================
def quaternion_to_yaw(orientation):
    siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
    cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
    return math.atan2(siny_cosp, cosy_cosp)

def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

# ======================== 无人机控制类 ========================
class UAVController:
    def __init__(self):
        self.pose = None
        self.image = None
        self.bridge = CvBridge()

        rospy.Subscriber('/ground_truth/state', Odometry, self.pose_callback)
        rospy.Subscriber('/front_cam/camera/image', Image, self.image_callback)

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.result_pub = rospy.Publisher('/tello/target_result', String, queue_size=10)

        self.linear_speed = 0.8
        self.angular_speed = 0.8

        rospy.loginfo("Waiting for pose...")
        while self.pose is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Pose received")

    def pose_callback(self, msg):    # 提取位置和偏航角
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        self.pose = (x, y, z, yaw)

    def image_callback(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            pass

    def stop(self):    # 停止运动
        self.cmd_pub.publish(Twist())

    def turn_to(self, target_yaw, tol=0.05, timeout=5):    # 转向，旋转到角度target_yaw
        if self.pose is None:
            return False
        rate = rospy.Rate(20)
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.logwarn("turn_to timeout")
                return False
            err = normalize_angle(target_yaw - self.pose[3])
            if abs(err) < tol:
                self.stop()
                return True
            wz = max(min(err * 1.5, self.angular_speed), -self.angular_speed)
            twist = Twist()
            twist.angular.z = wz
            self.cmd_pub.publish(twist)
            rate.sleep()
        return False

    def goto_xy(self, target_x, target_y, tol=0.2, timeout=30):    # 沿两点直线，在xy平面移动，
        if self.pose is None:
            rospy.logwarn("No pose yet")
            return False
        dx = target_x - self.pose[0]
        dy = target_y - self.pose[1]
        target_yaw = math.atan2(dy, dx)
        if not self.turn_to(target_yaw, tol=0.1):
            return False

        rate = rospy.Rate(20)
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.logwarn("goto_xy timeout")
                return False
            dx = target_x - self.pose[0]
            dy = target_y - self.pose[1]
            dist = math.hypot(dx, dy)
            if dist < tol:
                self.stop()
                return True
            current_target_yaw = math.atan2(dy, dx)
            if abs(normalize_angle(current_target_yaw - self.pose[3])) > 0.2:
                self.turn_to(current_target_yaw, tol=0.1)
            forward_speed = min(dist, self.linear_speed)
            twist = Twist()
            twist.linear.x = forward_speed
            self.cmd_pub.publish(twist)
            rate.sleep()
        return False

    def set_z(self, target_z, tol=0.1, timeout=10):    # 高度设置
        if self.pose is None:
            return False
        rate = rospy.Rate(20)
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.logwarn("set_z timeout")
                return False
            dz = target_z - self.pose[2]
            if abs(dz) < tol:
                self.stop()
                return True
            vz = max(min(dz * 1.0, self.linear_speed), -self.linear_speed)
            twist = Twist()
            twist.linear.z = vz
            self.cmd_pub.publish(twist)
            rate.sleep()
        return False

    # ------------------- 图像处理-火窗 -------------------
    def detect_fire(self, image):
        if image is None:
            return False
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius == 0:
                continue
            circularity = area / (math.pi * radius * radius)
            if circularity > 0.7:
                return True
        return False

    def enhance_shadow_region(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def detect_ball_at_position(self, image, position_id):
        """
        根据位置 ID 采用不同 HSV 阈值和筛选条件，
        解决位置3紫色地毯误检蓝色球、位置1/2/4阴影漏检蓝色球的问题。
        """
        if image is None:
            return 'e'

        # ========== 位置相关参数配置（阈值必须为 numpy 数组） ==========
        pos_configs = {
            # 位置1（阴影柜子高处）
            0: {
                'red':   [(np.array([0, 80, 60]), np.array([10, 255, 255])),
                          (np.array([160, 80, 60]), np.array([180, 255, 255]))],
                'yellow':[(np.array([20, 70, 70]), np.array([35, 255, 255]))],
                'blue':  [(np.array([95, 60, 40]), np.array([130, 255, 255]))],
                'min_area': 120,
                'circ_thresh': 0.55,
                'purple_suppress': False
            },
            # 位置2（阴影柜子第二层）
            1: {
                'red':   [(np.array([0, 100, 70]), np.array([10, 255, 255])),
                          (np.array([160, 100, 70]), np.array([180, 255, 255]))],
                'yellow':[(np.array([20, 70, 70]), np.array([35, 255, 255]))],
                'blue':  [(np.array([95, 60, 40]), np.array([130, 255, 255]))],
                'min_area': 150,
                'circ_thresh': 0.65,
                'purple_suppress': False
            },
            # 位置3（咖啡桌，紫色地毯干扰）
            2: {
                'red':   [(np.array([0, 140, 100]), np.array([10, 255, 255])),
                          (np.array([160, 140, 100]), np.array([180, 255, 255]))],
                'yellow':[(np.array([22, 100, 80]), np.array([35, 255, 255]))],
                'blue':  [(np.array([100, 120, 70]), np.array([125, 255, 255]))],
                'min_area': 250,
                'circ_thresh': 0.70,
                'purple_suppress': True
            },
            # 位置4（阴影柜子第四层）
            3: {
                'red':   [(np.array([0, 80, 60]), np.array([10, 255, 255])),
                          (np.array([160, 80, 60]), np.array([180, 255, 255]))],
                'yellow':[(np.array([20, 70, 70]), np.array([35, 255, 255]))],
                'blue':  [(np.array([95, 60, 40]), np.array([130, 255, 255]))],
                'min_area': 120,
                'circ_thresh': 0.55,
                'purple_suppress': False
            },
            # 位置5（墙角，光线正常）
            4: {
                'red':   [(np.array([0, 60, 40]), np.array([10, 255, 255])),
                          (np.array([158, 50, 40]), np.array([180, 255, 255]))],
                'yellow':[(np.array([20, 100, 80]), np.array([35, 255, 255]))],
                'blue':  [(np.array([100, 100, 70]), np.array([130, 255, 255]))],
                'min_area': 50,
                'circ_thresh': 0.50,
                'purple_suppress': False,
            }
        }

        cfg = pos_configs.get(position_id, pos_configs[4])   # 默认用位置5配置

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), dtype=np.uint8)

        # 紫色掩码（用于位置3排除干扰）
        purple_mask = None
        if cfg['purple_suppress']:
            lower_purple = np.array([130, 50, 50])
            upper_purple = np.array([160, 255, 255])
            purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
            purple_mask = cv2.dilate(purple_mask, kernel, iterations=2)

        best_guess = 'e'
        max_conf = 0

        for color_char, ranges in [('r', cfg['red']), ('y', cfg['yellow']), ('b', cfg['blue'])]:
            mask = None
            for lower, upper in ranges:
                m = cv2.inRange(hsv, lower, upper)
                mask = m if mask is None else cv2.bitwise_or(mask, m)

            if mask is None:
                continue

            # 若启用紫色抑制，从掩码中扣除紫色区域
            if purple_mask is not None:
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(purple_mask))

            # 形态学处理
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < cfg['min_area']:
                    continue

                # 圆度计算
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)

                # 外接圆面积比
                (_, _), radius = cv2.minEnclosingCircle(cnt)
                if radius > 0:
                    ratio = area / (np.pi * radius * radius)
                else:
                    ratio = 0

                # 综合评分
                if circularity > cfg['circ_thresh'] and ratio > 0.5:
                    conf = circularity * ratio
                    if conf > max_conf:
                        max_conf = conf
                        best_guess = color_char

        return best_guess

# ======================== 主任务状态机 ========================
class Mission:
    def __init__(self):
        self.uav = UAVController()
        self.state = "TAKEOFF"
        self.result_str = ['e'] * 5
        self.detected_window_x = 1.75
        self.target_positions = [
            (6.5, 7.0, 1.72),
            (5.0, 9.5, 1.0),
            (4.0, 11.0, 1.72),
            (1.0, 14.5, 0.2),
            (3.5, 7.5, 0.72)
        ]

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rospy.loginfo("State: %s", self.state)
            if self.state == "TAKEOFF":
                self.do_takeoff()
            elif self.state == "FIND_WINDOW":
                self.find_window()
            elif self.state == "CRUISE":
                self.cruise_and_detect()
            elif self.state == "LANDING":
                self.land()
            elif self.state == "FINISH":
                rospy.loginfo("Mission finished")
                break
            rate.sleep()

    def do_takeoff(self):    # 开始起飞
        rospy.loginfo("Taking off...")
        if self.uav.set_z(1.5, timeout=5):
            self.state = "FIND_WINDOW"
        else:
            rospy.logerr("Takeoff failed")
            rospy.signal_shutdown("Takeoff failed")

    def find_window(self):    # 检测并穿过窗口
        rospy.loginfo("Searching for fire mark...")
        self.uav.set_z(2.05)
        window_centers = [1.75, 4.25, 6.75]    # 窗中心x坐标
        for wx in window_centers:
            rospy.loginfo("Checking window at x=%.2f", wx)
            self.uav.goto_xy(wx, 2.5)    # 依次遍历
            self.uav.turn_to(math.pi/2)    # 转向y+方向
            rospy.sleep(0.5)
            if self.uav.image is not None and self.uav.detect_fire(self.uav.image):    # 正确查找
                rospy.loginfo("Fire found at window x=%.2f", wx)
                self.detected_window_x = wx
                rospy.loginfo("Crossing window at x=%.2f", self.detected_window_x)           
                self.uav.turn_to(math.pi/2)    # 确保机头朝前（y+）
                self.uav.set_z(1.25)    # 调整高度到窗户中间
                while self.uav.pose is not None and self.uav.pose[1] < 3.2:
                    self.uav.goto_xy(wx, 4.0)
                    rospy.sleep(0.1)
                self.uav.stop()
                rospy.loginfo("Inside the building")
                self.state = "CRUISE"               
                return
        # rospy.logwarn("No fire detected, using first window")     # 未探查到火窗
        # self.detected_window_x = 1.75
        # self.state = "CRUISE"

    def cruise_and_detect(self):
        rospy.loginfo("Starting object detection...")
        
        # 结果数组，索引0~4对应位置1~5
        self.result_str = ['e'] * 5

        # ① 飞到 (7.7, 4) → (7.7, 8.5) → (6.5, 8.5)，高度 2.05，检索位置1（索引0）
        rospy.loginfo("Step 1: Go to (7.7,4) then (7.7,8.5),then (6.5,8.5) height 2.05, check position1")
        self.uav.goto_xy(7.2, 4.0)
        self.uav.goto_xy(7.2, 5.7)
        self.uav.goto_xy(5.1, 5.7)
        self.uav.goto_xy(5.1, 8.5)
        self.uav.goto_xy(6.5, 8.5)
        self.uav.set_z(2.05)
        self.uav.turn_to(3*math.pi/2)    # 摄像头朝向y-方向
        rospy.sleep(0.5)
        if self.uav.image is not None:
            res = self.uav.detect_ball_at_position(self.uav.image, 0)
            self.result_str[0] = res
            rospy.loginfo("Position1 -> %s", res)
        else:
            rospy.logwarn("No image at position1")

        # ② 飞到 (5, 8.5)，高度 1.3，检索位置3（索引2）
        rospy.loginfo("Step 2: Go to (5,7.5), height 1.3, check position3")
        self.uav.goto_xy(5.0, 8.5)
        self.uav.set_z(1.3)
        self.uav.turn_to(math.pi/2)    # 摄像头朝向y+方向
        rospy.sleep(0.5)
        if self.uav.image is not None:
            res = self.uav.detect_ball_at_position(self.uav.image, 2)
            self.result_str[2] = res
            rospy.loginfo("Position3 -> %s", res)
        else:
            rospy.logwarn("No image at position3")

        # ③ 飞到 (5,6) → (2.5,6) → (2.5,7.5)，高度 1.05，检索位置2（索引1）
        rospy.loginfo("Step 3: Go to (5,6) -> (2.5,6) -> (2.5,7.5), height 1.05, check position2")
        self.uav.goto_xy(5.0, 6.0)
        self.uav.goto_xy(2.5, 6.0)
        self.uav.goto_xy(2.5, 7.5)
        self.uav.set_z(1.05)
        self.uav.turn_to(0)    # 摄像头朝向x+方向
        rospy.sleep(0.5)
        if self.uav.image is not None:
            res = self.uav.detect_ball_at_position(self.uav.image, 1)
            self.result_str[1] = res
            rospy.loginfo("Position2 -> %s", res)
        else:
            rospy.logwarn("No image at position2")

        # ④ 飞到 (2.5,12.5) → (4,12.5)，高度 2.05，检索位置4（索引3）
        rospy.loginfo("Step 4: Go to (2.5,12.5) -> (4,12.5), height 2.05, check position4")
        self.uav.goto_xy(2.5, 12.5)
        self.uav.goto_xy(4.0, 12.5)
        self.uav.set_z(2.05)
        self.uav.turn_to(3*math.pi/2)    # 摄像头朝向y-方向
        rospy.sleep(0.5)
        if self.uav.image is not None:
            res = self.uav.detect_ball_at_position(self.uav.image, 3)
            self.result_str[3] = res
            rospy.loginfo("Position4 -> %s", res)
        else:
            rospy.logwarn("No image at position4")

        # ⑤ 飞到 (4,14.5) → (2,14.5)，高度 0.55，检索位置5（索引4）
        rospy.loginfo("Step 5: Go to (4,14.5) -> (1.5,14.5), height 0.5, check position5")
        self.uav.goto_xy(4.0, 14.5)
        self.uav.goto_xy(2, 14.5)
        self.uav.set_z(0.2)
        self.uav.turn_to(math.pi)    # 摄像头朝向x-方向
        rospy.sleep(0.5)
        if self.uav.image is not None:
            res = self.uav.detect_ball_at_position(self.uav.image, 4)
            self.result_str[4] = res
            rospy.loginfo("Position5 -> %s", res)
        else:
            rospy.logwarn("No image at position5")

        self.uav.set_z(2.00)

        # 发送最终结果
        result_msg = ''.join(self.result_str)
        rospy.loginfo("Final result: %s", result_msg)
        self.uav.result_pub.publish(String(data=result_msg))
        rospy.sleep(1)
        self.state = "LANDING"

    def land(self):
        rospy.loginfo("Landing...")
        self.uav.goto_xy(3.5, 14.5)
        self.uav.goto_xy(3.5, 12.5)
        self.uav.goto_xy(7, 12.5)
        self.uav.goto_xy(7, 14.5)
        self.uav.set_z(0.2)
        self.uav.stop()
        self.state = "FINISH"


if __name__ == "__main__":
    rospy.init_node('uav_competition')
    mission = Mission()
    mission.run()
    