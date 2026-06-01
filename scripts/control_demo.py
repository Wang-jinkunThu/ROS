#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge


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

        self.lock = threading.Lock()

        # ----------------------------
        # 订阅话题
        # ----------------------------
        rospy.Subscriber(f"{name}/image_raw", Image, self.image_callback)
        rospy.Subscriber(f"{name}/pose", PoseStamped, self.pose_callback)

    # ---- 四元数转 yaw ----
    def quat_to_yaw(self, q):
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        siny_cosp = 2 * (qw*qz + qx*qy)
        cosy_cosp = 1 - 2 * (qy*qy + qz*qz)
        return math.degrees(math.atan2(siny_cosp, cosy_cosp))

    # ---- 图像更新 ----
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.image = frame

        cv2.imshow("Tello Image", frame)
        cv2.waitKey(1)

    # ---- 位姿更新 ----
    def pose_callback(self, msg: PoseStamped):
        yaw_deg = self.quat_to_yaw(msg.pose.orientation)

        with self.lock:
            self.position = msg.pose.position
            self.orientation = msg.pose.orientation
            self.yaw = yaw_deg

# ============================================================
#  TelloControl：控制 + 发布裁判机消息
# ============================================================
class TelloControl:
    def __init__(self, name):
        rospy.loginfo(f"Tello Name: {name}")
        self.state = TelloState(name)
        # ----------------------------
        # 控制话题
        # ----------------------------
        self.pub = rospy.Publisher(f"{name}/sdk_cmd", String, queue_size=1)

        # ----------------------------
        # 裁判机通信的发布器
        # ----------------------------
        self.judge_pub = rospy.Publisher(f"{name}/judge", String, queue_size=1)

        # 1. 信息打印示例
        self.ctrl_thread = threading.Thread(target=self.state_callback_demo)
        self.ctrl_thread.daemon = True
        self.ctrl_thread.start()

        # 2. 飞向目标位置的控制示例：飞往（0，2），机头朝前，注意必须在定位毯上才能执行
        self.target_x = 0.0
        self.target_y = -1.5
        self.target_z = 1.5
        self.target_yaw = 90.0
        self.ctrl_thread = threading.Thread(target=self.position_control_demo)
        self.ctrl_thread.daemon = True
        self.ctrl_thread.start()

    # ========== 基本控制指令 ==========
    def send(self, cmd):
        self.pub.publish(cmd)

    def forward(self, cm): self.send(f"forward {cm}")
    def back(self, cm): self.send(f"back {cm}")
    def left(self, cm): self.send(f"left {cm}")
    def right(self, cm): self.send(f"right {cm}")
    def up(self, cm): self.send(f"up {cm}")
    def down(self, cm): self.send(f"down {cm}")
    def cw(self, deg): self.send(f"cw {deg}")
    def ccw(self, deg): self.send(f"ccw {deg}")
    def stop(self): self.send("stop")
    def takeoff(self): self.send("takeoff")
    def land(self): self.send("land")

    def position_control_demo(self):
        # 控制 demo：前往目标位置，调整位姿
        rate = rospy.Rate(1)  # 控制频率 1 Hz
        print("=== 前往目标位置 Demo 已启动 ===")
        rospy.sleep(2)

        self.takeoff()
        rospy.sleep(5)

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
                if yaw_err > 0: 
                    self.ccw(10)
                    print("yaw err: ", yaw_err, ", ccw 10")
                else:           
                    self.cw(10)
                    print("yaw err: ", yaw_err, ", cw 10")
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
                # ----------------------------
                # 裁判机发送示例：到达目标（决赛中可以参考这个发送识别结果）
                # ----------------------------
                self.judge_pub.publish("reach_target")
                rate.sleep()
                continue

            # X 控制
            if abs(ex) >= 0.2:
                if ex > 0: 
                    self.right(int(20))
                    print("x err: ", ex, ", right 20")
                else:      
                    self.left(int(20))
                    print("x err: ", ex, ", left 20")
                continue

            # Y 控制
            if abs(ey) >= 0.2:
                if ey > 0: 
                    self.forward(int(20))
                    print("y err: ", ey, ", forward 20")
                else:      
                    self.back(int(20))
                    print("y err: ", ey, ", back 20")
                continue

            # z 控制
            if abs(ez) >= 0.2:
                if ez > 0: 
                    self.up(20)
                    print("z err: ", ez, ", up 20")
                else:      
                    self.down(20)
                    print("z err: ", ez, ", down 20")
                continue
            
            rate.sleep()

    def state_callback_demo(self):
        """
        打印位姿 + 显示图像的例程
        """
        rate = rospy.Rate(5)  # 5Hz 刷新频率

        print("=== 状态回调 Demo 已启动 ===")

        while not rospy.is_shutdown():

            # -----------------------------
            # 等待状态
            # -----------------------------
            if self.state.position is None or self.state.yaw is None:
                rate.sleep()
                continue

            # -----------------------------
            # 读取位姿
            # -----------------------------
            with self.state.lock:
                px = self.state.position.x
                py = self.state.position.y
                yaw = self.state.yaw
                frame = self.state.image  # 当前图像

            # -----------------------------
            # 打印状态
            # -----------------------------
            print(f"[State] x={px:.2f}, y={py:.2f}, yaw={yaw:.2f} deg")

            # -----------------------------
            # 显示图像（如果有）
            # -----------------------------
            if frame is not None:
                cv2.imshow("Tello State Demo", frame)
                cv2.waitKey(1)

            # -----------------------------
            # 裁判机通信示例
            # -----------------------------
            self.judge_pub.publish("state_ok")

            rate.sleep()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    rospy.init_node("tello_control_node", anonymous=False)
    name = rospy.get_param('~name', "")
    control = TelloControl(name)
    control.position_control_demo()
