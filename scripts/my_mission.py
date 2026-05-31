#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_srvs.srv import SetBool
from Tello import TelloControl


# ============================================================
#  StateMachine：主任务状态机
# ============================================================
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
                elif self.state == "DETECT_BALL":
                    self.do_detectball()
                elif self.state == "FIND_WINDOW":
                    self.do_window()
                elif self.state == "DETECT_LIGHT":
                    self.do_light()
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
                self.state = "FINISH"
                self.uav.land()

            rate.sleep()

    # ========== 各状态实现 ==========
    def do_takeoff(self):
        # 记录起飞前高度
        init_z = 0.0
        for _ in range(10):
            with self.uav.state.lock:
                if self.uav.state.position is not None:
                    init_z = self.uav.state.position.z
                    break
            rospy.sleep(0.3)
        rospy.loginfo(f"Initial height: {init_z:.2f}m")

        rospy.loginfo("Taking off...")
        self.uav.takeoff()

        # 等待高度上升 0.3m 以上确认起飞
        deadline = rospy.Time.now() + rospy.Duration(15)
        while not rospy.is_shutdown():
            with self.uav.state.lock:
                pz = self.uav.state.position.z if self.uav.state.position else init_z
            if pz - init_z > 0.3:
                rospy.loginfo(f"Takeoff confirmed, height={pz:.2f}m (delta={pz - init_z:.2f}m)")
                break
            if rospy.Time.now() > deadline:
                rospy.logerr(f"Takeoff timeout: delta={pz - init_z:.2f}m")
                self.state = "LANDING"
                return
            rospy.sleep(0.5)

        # 起飞后稍等让飞控稳定，再启用下视视觉定位
        rospy.sleep(2)

        drone_name = self.uav.state.name
        service_name = f"/{drone_name}/set_downvision" if drone_name else "/set_downvision"
        rospy.loginfo(f"Waiting for service: {service_name}")
        rospy.wait_for_service(service_name)

        downvision_ok = False
        for attempt in range(3):
            try:
                set_down = rospy.ServiceProxy(service_name, SetBool)
                resp = set_down(True)
                if resp.success:
                    rospy.loginfo("Downvision enabled successfully")
                    downvision_ok = True
                    break
                else:
                    rospy.logwarn(f"Downvision attempt {attempt + 1}/3 failed: {resp.message}")
                    rospy.sleep(1)
            except Exception as e:
                rospy.logwarn(f"Downvision attempt {attempt + 1}/3 exception: {e}")
                rospy.sleep(1)

        if not downvision_ok:
            rospy.logerr("Downvision enable failed after 3 attempts")
            self.state = "LANDING"
            return

        # 等待定位数据就绪
        rospy.loginfo("Waiting for position data...")
        deadline = rospy.Time.now() + rospy.Duration(10)
        while not rospy.is_shutdown():
            with self.uav.state.lock:
                ok = self.uav.state.position is not None and self.uav.state.yaw is not None
            if ok:
                rospy.loginfo("Position data ready")
                break
            if rospy.Time.now() > deadline:
                rospy.logerr("Timeout waiting for position data")
                self.state = "LANDING"
                return
            rospy.sleep(0.5)

        self.state = _

    def do_detectball(self):
        if not self.uav.goto_xy(150, 125):
            rospy.logerr("Failed to reach rotating ball observation point")
            self.state = "LANDING"
            return
        if not self.uav.move_z(0):
            rospy.logerr("Failed to set height for rotating ball")
            self.state = "LANDING"
            return
        if not self.uav.turn_to(0):
            rospy.logerr("Failed to turn to rotating ball")
            self.state = "LANDING"
            return

        ball1_color = self.uav.wait_for_ball(timeout=10)
        if ball1_color is None:
            rospy.logwarn("Rotating ball not detected, set to unknown")
            ball1_color = '?'
        rospy.loginfo(f"Rotating ball color: {ball1_color}")

        for pt in [(150, 50), (200, 50), (310, 125)]:
            if not self.uav.goto_xy(*pt):
                rospy.logerr(f"Failed to reach waypoint {pt}")
                self.state = "LANDING"
                return

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
        self.state = "FIND_WINDOW"

    def do_window(self):
        detect_points = [(250, 235), (150, 235), (350, 235)]
        fire_found = False
        selected_x = None
        self.uav.move_z(105)

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

        self.uav.move_z(-27)
        self.uav.goto_xy(selected_x, 235)

        rospy.sleep(2)
        self.state = "DETECT_LIGHT"

    def do_light(self):
        for i in range(3):
            self.uav.goto_xy(250, 400)
            self.uav.move_z(-23)
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
            with self.uav.state.lock:
                cur_z = self.uav.state.position.z * 100.0 if self.uav.state.position else 0
            rospy.loginfo(f"Go to x={x_world}, z={z_world}")
            self.uav.goto_xy(x_world, 500)
            self.uav.move_z(z_world - cur_z)

            target_msg = f"target{i + 1}"
            self.uav.judge_pub.publish(target_msg)
            print(f"[JUDGE] Target sent: {target_msg}")
            rospy.sleep(1)

        rospy.sleep(2)
        self.state = "LANDING"

    def do_landing(self):
        rospy.loginfo("Landing...")
        self.uav.goto_xy(470, 470)
        with self.uav.state.lock:
            cur_z = self.uav.state.position.z * 100.0 if self.uav.state.position else 0
        self.uav.move_z(30 - cur_z)
        self.uav.land()
        rospy.sleep(3)
        self.state = "FINISH"


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    rospy.init_node("tello_control_node", anonymous=False)
    name = rospy.get_param('~name', "")
    my_mission = StateMachine(name)
    my_mission.run()
