#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：起飞后输入 x/y 方向相对移动距离（cm），无人机水平移动到位。
测试 Tello.py 中的 move_x() 和 move_y()。
"""

import rospy
from Tello import TelloControl


def main():
    rospy.init_node("test_move_xy_node", anonymous=False)
    name = rospy.get_param('~name', "")
    uav = TelloControl(name)

    # --- 起飞 ---
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
    rospy.sleep(2)

    # --- 交互循环 ---
    print("\n" + "=" * 50)
    print("  输入方向和距离（cm），输入 q 退出")
    print("  格式: x 30   → 右移 30cm")
    print("        x -30  → 左移 30cm")
    print("        y 50   → 前进 50cm")
    print("        y -50  → 后退 50cm")
    print("=" * 50 + "\n")

    while not rospy.is_shutdown():
        try:
            raw = input("move > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if raw.lower() in ("q", "quit", "exit"):
            break
        if not raw:
            continue

        parts = raw.split()
        if len(parts) != 2:
            print("格式错误，示例: x 30 或 y -50")
            continue

        axis, val = parts[0].lower(), parts[1]
        try:
            delta = int(val)
        except ValueError:
            print(f"无效数字: '{val}'")
            continue

        if axis == "x":
            ok = uav.move_x(delta)
        elif axis == "y":
            ok = uav.move_y(delta)
        else:
            print(f"未知方向: '{axis}'，请输入 x 或 y")
            continue

        with uav.state.lock:
            if uav.state.position:
                px = uav.state.position.x * 100.0
                py = uav.state.position.y * 100.0
            else:
                px = py = 0
        if ok:
            print(f"→ 到位! 当前坐标: x={px:.1f} y={py:.1f} cm")
        else:
            print(f"→ 超时未到位 当前坐标: x={px:.1f} y={py:.1f} cm")

    # --- 降落 ---
    rospy.loginfo("Landing...")
    uav.land()
    rospy.sleep(3)


if __name__ == "__main__":
    main()
