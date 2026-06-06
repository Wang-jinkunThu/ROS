#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：按照 path.md 规划的路径完成完整任务。
调用 Tello.py 中的控制方法，参考 my_mission.py 的状态机结构。
goto_xy() 已改写为先 x 后 y 的移动方式，不再使用 turn_to()。
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

            assert init_z is not None
            assert pz is not None
            if pz - init_z > 0.3:
                rospy.loginfo(f"Takeoff confirmed, height={pz:.2f}m")
                break
            if rospy.Time.now() > deadline:
                rospy.logerr("Takeoff confirmation timeout")
                self.state = "LANDING"
                return
            rospy.sleep(0.5)

        rospy.sleep(2)

        # 上升到 53cm
        rospy.loginfo("Rising to 53cm...")
        if not self.uav.set_z(53):
            rospy.logerr("Failed to reach 53cm")
            self.state = "LANDING"
            return

        self.state = "ROTATING_BALL"

    def do_rotating_ball(self):
        # 飞向旋转柜观察点 (-102, -181)，高度 53cm
        rospy.loginfo("Going to rotating ball observation point (-102, -181)...")
        if not self.uav.goto_xy(-102, -181):
            rospy.logerr("Failed to reach rotating ball point")
            self.state = "LANDING"
            return

        # 逆时针旋转 90 度，面朝旋转柜
        rospy.loginfo("Turning CCW 90 deg to face rotating cabinet...")
        with self.uav.state.lock:
            cur_yaw = self.uav.state.yaw or 0
        if not self.uav.turn_to(cur_yaw + 90, tol_deg=10, timeout=20):
            rospy.logerr("Failed to turn to rotating cabinet")
            self.state = "LANDING"
            return

        # 悬停 10 秒检测球
        rospy.loginfo("Hovering 3s, detecting ball color...")
        rospy.sleep(3)
        ball1 = self.uav.wait_for_ball(timeout=10)
        if ball1 is None:
            rospy.logwarn("Rotating ball not detected, set to unknown")
            ball1 = '?'
        rospy.loginfo(f"Rotating ball color: {ball1}")

        self.ball1 = ball1
        self.state = "FIXED_BALL"

    def do_fixed_ball(self):
        # 飞向固定柜观察点序列（保持 z=53 不变）
        waypoints = [(60, -181), (60, -120)]
        for wx, wy in waypoints:
            rospy.loginfo(f"Going to ({wx}, {wy})...")
            if not self.uav.goto_xy(wx, wy):
                rospy.logerr(f"Failed to reach ({wx}, {wy})")
                self.state = "LANDING"
                return

        # 上升到 118cm
        if not self.uav.set_z(118):
            rospy.logerr("Failed to reach 118cm")
            self.state = "LANDING"
            return

        # 顺时针旋转 90 度，面朝固定柜
        rospy.loginfo("Turning CW 90 deg to face fixed cabinet...")
        with self.uav.state.lock:
            cur_yaw = self.uav.state.yaw or 90
        if not self.uav.turn_to(cur_yaw - 90, tol_deg=10, timeout=20):
            rospy.logerr("Failed to turn to fixed cabinet")
            self.state = "LANDING"
            return

        # 悬停 10 秒检测球
        rospy.loginfo("Hovering 10s, detecting ball color...")
        rospy.sleep(3)
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

        self.state = "FIND_WINDOW"

    def do_find_window(self):
        # 飞向窗户区域
        approach_points = [(60, -16), (66, -16)]
        for wx, wy in approach_points:
            rospy.loginfo(f"Going to ({wx}, {wy})...")
            if not self.uav.goto_xy(wx, wy):
                rospy.logerr(f"Failed to reach ({wx}, {wy})")
                self.state = "LANDING"
                return

        # 上升到 192cm 检查窗户
        if not self.uav.set_z(192):
            rospy.logerr("Failed to reach 192cm")
            self.state = "LANDING"
            return

        # 依次检查三个窗户：从右到左
        window_points = [(66, -16), (0, 0), (-107, -23)]
        window_heights = [192, 189, 188]
        fire_found = False

        for (wx, wy), wz in zip(window_points, window_heights):
            rospy.loginfo(f"Checking window at ({wx}, {wy}), height={wz}cm...")
            if not self.uav.goto_xy(wx, wy):
                rospy.logwarn(f"Failed to reach ({wx}, {wy}), skip")
                continue

            if not self.uav.set_z(wz):
                rospy.logwarn(f"Failed to set height {wz}cm")
                continue

            rospy.sleep(1)
            if self.uav.detect_fire():
                rospy.loginfo(f"Fire window detected at ({wx}, {wy})")
                fire_found = True
                break
            rospy.loginfo(f"No fire at ({wx}, {wy})")

        # 检测到红点：先 z 到 152，再 y 方向 +50
        if fire_found:
            rospy.loginfo("Fire detected: setting z=152 then moving y+50...")
            if not self.uav.set_z(152):
                rospy.logwarn("set_z(152) failed")
            if not self.uav.move_y(50):
                rospy.logwarn("move_y(50) failed")

        # 穿过窗户后统一移动到 (0, 50, 163)
        rospy.loginfo("Going to gathering point (0, 50)...")
        if not self.uav.goto_xy(0, 50):
            rospy.logerr("Failed to reach gathering point (0, 50)")
            self.state = "LANDING"
            return

        if not self.uav.set_z(163):
            rospy.logerr("Failed to reach 163cm")
            self.state = "LANDING"
            return

        self.state = "DETECT_LIGHT"

    def do_detect_light(self):
        # 逆时针旋转 90 度，面朝白板
        rospy.loginfo("Turning CCW 90 deg to face whiteboard...")
        with self.uav.state.lock:
            cur_yaw = self.uav.state.yaw or 0
        if not self.uav.turn_to(cur_yaw + 90, tol_deg=10, timeout=20):
            rospy.logerr("Failed to turn to whiteboard")
            self.state = "LANDING"
            return

        rospy.sleep(1)

        # 模板匹配检测白板 3×3 网格，获取 LIGHT_ON 的世界坐标
        rospy.loginfo("Detecting whiteboard grid via template matching...")
        result = self.uav.detect_whiteboard_light(timeout=15)
        if result is None:
            rospy.logerr("Cannot detect whiteboard LIGHT_ON")
            self.state = "LANDING"
            return

        target_x, target_y, target_z = result
        rospy.loginfo(f"LIGHT_ON target: x={target_x:.1f}, y={target_y:.1f}, z={target_z:.1f}")

        # 飞向目标位置
        rospy.loginfo(f"Going to target ({target_x:.1f}, {target_y:.1f}, {target_z:.1f})...")
        if not self.uav.goto_xy(target_x, target_y):
            rospy.logerr("Failed to reach target position")
            self.state = "LANDING"
            return

        if not self.uav.set_z(target_z):
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
