#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：按照 path.md 规划的路径完成完整任务。
调用 Tello.py 中的控制方法，参考 my_mission.py 的状态机结构。
"""

import rospy
from Tello import TelloControl


class StateMachine:
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
                elif self.state == "ROTATING_BALL":
                    self.do_rotating_ball()
                elif self.state == "FIXED_BALL":
                    self.do_fixed_ball()
                elif self.state == "FIND_WINDOW":
                    self.do_find_window()
                elif self.state == "DETECT_LIGHT":
                    self.do_detect_light()
                elif self.state == "LANDING":
                    self.do_landing()
                elif self.state == "FINISH":
                    rospy.loginfo("Mission finished")
                    break
                else:
                    rospy.logerr(f"Unknown state: {self.state}")
                    break
            except Exception as e:
                rospy.logerr(f"Exception in state '{self.state}': {e}")
                rospy.loginfo("Emergency landing...")
                self.uav.land()
                self.state = "FINISH"

            rate.sleep()

    # ========== 各状态实现 ==========
    def do_takeoff(self):
        rospy.sleep(2)
        rospy.loginfo("Taking off...")
        self.uav.takeoff()

        # 等待 pose 数据出现（无人机运动后才发布）
        rospy.loginfo("Waiting for position data...")
        deadline = rospy.Time.now() + rospy.Duration(15)
        init_z = None
        while not rospy.is_shutdown():
            with self.uav.state.lock:
                if self.uav.state.position is not None:
                    init_z = self.uav.state.position.z
                    break
            if rospy.Time.now() > deadline:
                rospy.logerr("Timeout waiting for position data after takeoff")
                self.state = "LANDING"
                return
            rospy.sleep(0.5)

        rospy.loginfo(f"Position received, height={init_z:.2f}m")

        # 确认起飞：高度上升 0.3m 以上
        rospy.loginfo("Confirming takeoff...")
        deadline = rospy.Time.now() + rospy.Duration(15)
        while not rospy.is_shutdown():
            with self.uav.state.lock:
                pz = self.uav.state.position.z if self.uav.state.position else init_z
            if pz - init_z > 0.3:
                rospy.loginfo(f"Takeoff confirmed, height={pz:.2f}m")
                break
            if rospy.Time.now() > deadline:
                rospy.logerr("Takeoff confirmation timeout")
                self.state = "LANDING"
                return
            rospy.sleep(0.5)

        rospy.sleep(2)

        # 上升到 40cm
        rospy.loginfo("Rising to 40cm...")
        if not self.uav.set_z(40, tol_cm=20, timeout=15):
            rospy.logerr("Failed to reach 40cm")
            self.state = "LANDING"
            return

        self.state = "ROTATING_BALL"

    def do_rotating_ball(self):
        # 飞向旋转柜观察点
        rospy.loginfo("Going to rotating ball observation point (-104, -130)...")
        if not self.uav.goto_xy(-104, -130, tol_cm=20, timeout=30):
            rospy.logerr("Failed to reach rotating ball point")
            self.state = "LANDING"
            return

        # 逆时针旋转 90 度，面朝旋转柜
        rospy.loginfo("Turning CCW 90 deg to face rotating cabinet...")
        with self.uav.state.lock:
            cur_yaw = self.uav.state.yaw or 0
        target_yaw = cur_yaw + 90
        if not self.uav.turn_to(target_yaw, tol_deg=10, timeout=20):
            rospy.logerr("Failed to turn to rotating cabinet")
            self.state = "LANDING"
            return

        # 悬停 10 秒检测球
        rospy.loginfo("Hovering 10s, detecting ball color...")
        rospy.sleep(10)
        ball1 = self.uav.wait_for_ball(timeout=10)
        if ball1 is None:
            rospy.logwarn("Rotating ball not detected, set to unknown")
            ball1 = '?'
        rospy.loginfo(f"Rotating ball color: {ball1}")

        self.ball1 = ball1
        self.state = "FIXED_BALL"

    def do_fixed_ball(self):
        # 飞向固定柜观察点序列
        waypoints = [(0, -130), (0, -125), (30, -125)]
        for wx, wy in waypoints:
            rospy.loginfo(f"Going to ({wx}, {wy})...")
            if not self.uav.goto_xy(wx, wy, tol_cm=20, timeout=30):
                rospy.logerr(f"Failed to reach ({wx}, {wy})")
                self.state = "LANDING"
                return

        # 顺时针旋转 90 度，面朝固定柜
        rospy.loginfo("Turning CW 90 deg to face fixed cabinet...")
        with self.uav.state.lock:
            cur_yaw = self.uav.state.yaw or 90
        target_yaw = cur_yaw - 90
        if not self.uav.turn_to(target_yaw, tol_deg=10, timeout=20):
            rospy.logerr("Failed to turn to fixed cabinet")
            self.state = "LANDING"
            return

        # 悬停 10 秒检测球
        rospy.loginfo("Hovering 10s, detecting ball color...")
        rospy.sleep(10)
        ball2 = self.uav.wait_for_ball(timeout=10)
        if ball2 is None:
            rospy.logwarn("Fixed ball not detected, set to unknown")
            ball2 = '?'
        rospy.loginfo(f"Fixed ball color: {ball2}")

        # 发送裁判机结果
        ball_result = f"{self.ball1}{ball2}"
        rospy.loginfo(f"Ball detection result: {ball_result}")
        self.uav.judge_pub.publish(ball_result)
        print(f"[JUDGE] Ball result sent: {ball_result}")

        # 上升至 163cm
        rospy.loginfo("Rising to 163cm...")
        if not self.uav.set_z(163, tol_cm=20, timeout=20):
            rospy.logerr("Failed to reach 163cm")
            self.state = "LANDING"
            return

        self.state = "FIND_WINDOW"

    def do_find_window(self):
        # 先飞到窗口区域
        rospy.loginfo("Going to (30, -20)...")
        if not self.uav.goto_xy(30, -20, tol_cm=20, timeout=30):
            rospy.logerr("Failed to reach (30, -20)")
            self.state = "LANDING"
            return

        # 依次检查三个窗户：从右到左 (x=100, 0, -100)
        window_points = [(100, -20), (0, -20), (-100, -20)]
        fire_found = False
        fire_wx = None

        for wx, wy in window_points:
            rospy.loginfo(f"Checking window at ({wx}, {wy})...")
            if not self.uav.goto_xy(wx, wy, tol_cm=20, timeout=30):
                rospy.logwarn(f"Failed to reach ({wx}, {wy}), skip")
                continue

            rospy.sleep(1)
            if self.uav.detect_fire():
                rospy.loginfo(f"Fire window detected at ({wx}, {wy})")
                fire_found = True
                fire_wx = wx
                break
            rospy.loginfo(f"No fire at ({wx}, {wy})")

        # 检测到红点则 y 正方向移动 50cm
        if fire_found:
            rospy.loginfo("Moving +50cm in y direction...")
            if not self.uav.move_y(50, tol_cm=20, timeout=15):
                rospy.logwarn("move_y +50 failed, continuing")

        # 统一移动到 (0, 50, 163)
        rospy.loginfo("Going to gathering point (0, 50)...")
        if not self.uav.goto_xy(0, 50, tol_cm=20, timeout=30):
            rospy.logerr("Failed to reach gathering point (0, 50)")
            self.state = "LANDING"
            return

        # 下降到 140cm
        rospy.loginfo("Descending to 140cm...")
        if not self.uav.set_z(140, tol_cm=20, timeout=15):
            rospy.logerr("Failed to reach 140cm")
            self.state = "LANDING"
            return

        self.state = "DETECT_LIGHT"

    def do_detect_light(self):
        # 逆时针旋转 90 度，面朝白板
        rospy.loginfo("Turning CCW 90 deg to face whiteboard...")
        with self.uav.state.lock:
            cur_yaw = self.uav.state.yaw or 0
        target_yaw = cur_yaw + 90
        if not self.uav.turn_to(target_yaw, tol_deg=10, timeout=20):
            rospy.logerr("Failed to turn to whiteboard")
            self.state = "LANDING"
            return

        rospy.sleep(1)

        # 检测白板角点
        rospy.loginfo("Detecting whiteboard corners...")
        corners = None
        deadline = rospy.Time.now() + rospy.Duration(30)
        while corners is None and not rospy.is_shutdown():
            corners = self.uav.detect_whiteboard_corners()
            if rospy.Time.now() > deadline:
                break
            rospy.sleep(0.5)

        if corners is None:
            rospy.logerr("Cannot detect whiteboard corners")
            self.state = "LANDING"
            return

        rospy.loginfo("Whiteboard corners detected")

        # 检测红灯
        rospy.loginfo("Detecting red light...")
        pixel = None
        deadline = rospy.Time.now() + rospy.Duration(30)
        while pixel is None and not rospy.is_shutdown():
            pixel = self.uav.detect_red_light()
            if rospy.Time.now() > deadline:
                break
            rospy.sleep(0.5)

        if pixel is None:
            rospy.logerr("No red light detected")
            self.state = "LANDING"
            return

        # 像素坐标 → 世界坐标
        x_world, z_world = self.uav.pixel_to_world(pixel[0], pixel[1], corners)
        rospy.loginfo(f"Red light world position: x={x_world:.1f}, z={z_world:.1f}")

        # 飞向目标位置
        rospy.loginfo(f"Going to target ({x_world:.1f}, 50)...")
        if not self.uav.goto_xy(x_world, 50, tol_cm=20, timeout=30):
            rospy.logerr("Failed to reach target position")
            self.state = "LANDING"
            return

        with self.uav.state.lock:
            cur_z = self.uav.state.position.z * 100.0 if self.uav.state.position else 140

        if not self.uav.set_z(z_world, tol_cm=20, timeout=15):
            rospy.logerr("Failed to set target height")
            self.state = "LANDING"
            return

        # 发送裁判机目标
        self.uav.judge_pub.publish("target")
        print("[JUDGE] Target sent")
        rospy.sleep(2)

        self.state = "LANDING"

    def do_landing(self):
        rospy.loginfo("Landing...")
        self.uav.land()
        rospy.sleep(3)
        self.state = "FINISH"


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    rospy.init_node("test_my_mission_node", anonymous=False)
    name = rospy.get_param('~name', "")
    mission = StateMachine(name)
    mission.run()
