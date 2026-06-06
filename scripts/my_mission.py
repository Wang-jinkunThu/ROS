#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import rospy
from Tello import TelloControl


class StateMachine:
    def __init__(self, drone_name):
        self.uav = TelloControl(drone_name)
        self.state = "TAKEOFF"

    def run(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            rospy.loginfo(f"Current state: {self.state}")

            try:
                if self.state == "TAKEOFF":
                    self.do_takeoff()
                elif self.state == "DETECT_BALL":
                    self.do_detect_boll()
                elif self.state == "FIND_WINDOW":
                    self.do_find_window()
                elif self.state == "DETECT_LIGHT":
                    self.do_detect_light()
                elif self.state == "LANDING":
                    self.do_landing()
                elif self.state == "FINISH":
                    self.do_finish()
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

        rospy.sleep(2)
        self.state = "DETECT_BALL"

    def do_detect_boll(self):
        # 飞向旋转柜观察点 (-102, -200)，高度 50cm，角度y+
        rospy.loginfo("Going to rotating ball observation point (-102, -181)...")
        self.uav.position_change(-1.02, -1.85, 0.5, 90)
        self.uav.position_control_demo()

        # 悬停 10 秒检测球
        rospy.loginfo("Hovering 10s, detecting ball color...")
        rospy.sleep(10)
        ball1 = self.uav.wait_for_ball(timeout=15)
        if ball1 is None:
            rospy.logwarn("Rotating ball not detected, set to unknown")
            ball1 = '?'
        rospy.loginfo(f"Rotating ball color: {ball1}")


        self.ball1 = ball1

        # 飞向固定柜观察点 (80, -180)，高度 150cm，角度x+
        self.uav.position_change(0.5, -1.35, 1.15, 90)
        self.uav.position_control_demo()
        self.uav.position_change(0.5, -1.35, 1.15, 0)
        self.uav.position_control_demo()

        # 悬停 10 秒检测球
        rospy.loginfo("Hovering 10s, detecting ball color...")
        rospy.sleep(10)
        ball2 = self.uav.wait_for_ball(timeout=5)
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

        # 依次检查三个窗户：从左到右
        window_position = [-1, 0.1, 1]
        fire_found = False

        for wx in window_position:
            rospy.loginfo(f"Checking window at ({wx}, -0.4, 1.85)...")
            self.uav.position_change(wx, -0.4, 1.85, 90)
            self.uav.position_control_demo()
            rospy.sleep(3)

            if self.uav.detect_fire(timeout = 2):
                rospy.loginfo(f"Fire window detected at ({wx}, -0.4, 1.85)")
                fire_found = True
                break
            rospy.loginfo(f"No fire at ({wx}, -0.4, 1.85)")

        # 穿过火窗
        if fire_found:
            self.uav.position_change(wx, -0.4, 1.5, 90) # 高度下降
            self.uav.position_control_demo()
            self.uav.position_change(wx, 0.4, 1.5, 90) # 穿过火窗
            self.uav.position_control_demo()
        else:
            self.state = "LANDING"

        self.state = "DETECT_LIGHT"

    def do_detect_light(self):
        rospy.sleep(1)
        # 移动到 (0.1, 50, 163),角度y+
        self.uav.position_change(0.1, 0.3, 1.63, 90)
        self.uav.position_control_demo()
        obs_x, obs_y, obs_z = 0.1, 0.3, 1.63
        
        # 依次检测 3 次红灯
        for light_num in range(1, 4):
            rospy.loginfo(f"=== 第 {light_num} 次检测红灯 ===")

            # 等待并检测 LIGHT_ON
            result = self.uav.detect_whiteboard_light(timeout=15)
            if result is None:
                rospy.logerr(f"第 {light_num} 次未检测到红灯")
                self.state = "LANDING"
                return

            target_x, target_y, target_z = result  # cm
            rospy.loginfo(f"LIGHT_ON target: ({target_x:.1f}, {target_y:.1f}, {target_z:.1f}) cm")

            # 飞向目标位置（cm → m）
            self.uav.position_change(target_x / 100, target_y / 100, target_z / 100 + 0.1, 90)
            self.uav.position_control_demo()

            # 发送裁判机目标
            self.uav.judge_pub.publish(f"target{light_num}")
            print(f"[JUDGE]Target{light_num}")
            rospy.sleep(1)

            # 回到观察点（最后一次不需要返回）
            if light_num < 3:
                rospy.loginfo(f"Returning to observation point...")
                self.uav.position_change(obs_x, obs_y, obs_z, 90)
                self.uav.position_control_demo()

        rospy.sleep(2)

        self.state = "FINISH"

    def do_landing(self):
        
        rospy.loginfo("Landing...")
        self.uav.land()
        rospy.sleep(3)
    
    def do_finish(self):
        rospy.loginfo("Landing...")
        self.uav.position_change(2.2, 2.2, 0.3, 90)
        self.uav.position_control_demo()
        self.uav.land()
        rospy.sleep(3)


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    rospy.init_node("test_my_mission_node", anonymous=False)
    name = rospy.get_param('~name', "")
    mission = StateMachine(name)
    mission.run()
