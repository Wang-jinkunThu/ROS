#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：起飞后输入相对高度变化量（cm），无人机升降到位。
测试 Tello.py 中的 move_z()。
"""

import rospy
from Tello import TelloControl


def main():
    rospy.init_node("test_move_z_node", anonymous=False)
    name = rospy.get_param('~name', "")
    uav = TelloControl(name)

    # --- 起飞（静止时无 pose 数据，先起飞再等 pose）---
    rospy.sleep(2)

    rospy.loginfo("Taking off...")
    uav.takeoff()

    rospy.loginfo("Waiting for position data...")
    deadline = rospy.Time.now() + rospy.Duration(15)
    init_z = None
    while not rospy.is_shutdown():
        with uav.state.lock:
            if uav.state.position is not None:
                init_z = uav.state.position.z
                break
        if rospy.Time.now() > deadline:
            rospy.logerr("Timeout waiting for position data after takeoff")
            uav.land()
            return
        rospy.sleep(0.5)

    assert init_z is not None
    rospy.loginfo(f"First position received, initial height: {init_z:.2f}m")

    deadline = rospy.Time.now() + rospy.Duration(10)
    while not rospy.is_shutdown():
        with uav.state.lock:
            pz = uav.state.position.z if uav.state.position else init_z
        if pz - init_z > 0.3:
            rospy.loginfo(f"Takeoff OK (height={pz:.2f}m, delta={pz - init_z:.2f}m)")
            break
        if rospy.Time.now() > deadline:
            rospy.logerr(f"Takeoff timeout: delta={pz - init_z:.2f}m")
            uav.land()
            return
        rospy.sleep(0.5)

    rospy.sleep(2)

    # --- 交互循环 ---
    print("\n" + "=" * 50)
    print("  输入相对高度变化量（cm），输入 q 退出")
    print("  正数上升，负数下降，例如: 20, -30, 50")
    print("=" * 50 + "\n")

    while not rospy.is_shutdown():
        try:
            raw = input("delta_cm > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if raw.lower() in ("q", "quit", "exit"):
            break
        if not raw:
            continue

        try:
            delta = int(raw)
        except ValueError:
            print(f"无效输入: '{raw}'，请输入整数")
            continue

        ok = uav.move_z(delta, tol_cm=5, timeout=15)
        with uav.state.lock:
            cur_z = uav.state.position.z * 100.0 if uav.state.position else 0
        if ok:
            print(f"→ 到位! 当前高度: {cur_z:.1f} cm")
        else:
            print(f"→ 超时未到位 当前高度: {cur_z:.1f} cm")

    # --- 降落 ---
    rospy.loginfo("Landing...")
    uav.land()
    rospy.sleep(3)


if __name__ == "__main__":
    main()
