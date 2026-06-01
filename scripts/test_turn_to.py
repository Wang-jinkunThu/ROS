#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：起飞后输入目标角度（90° 的整数倍），无人机旋转到位。
独立实现的 turn_to，不依赖 Tello.py 中的 turn_to()。
"""

import rospy
from Tello import TelloControl


def get_yaw(uav):
    """安全读取当前 yaw，失败返回 None"""
    with uav.state.lock:
        return uav.state.yaw


def normalize_180(deg):
    """将角度归一化到 [-180, 180)"""
    deg = deg % 360
    if deg > 180:
        deg -= 360
    return deg


def simple_turn_to(uav, target_deg, tol_deg=10, timeout=20):
    """
    两阶段旋转：
      粗调 — 大步长快速接近目标（max 30° per step）
      精调 — 小步长消除剩余误差（10° per step）
    成功返回 True，超时返回 False。
    """
    deadline = rospy.Time.now() + rospy.Duration(timeout)

    phase = "coarse"       # coarse → fine → done
    coarse_step = 30       # 粗调每步转 30°
    fine_step = 10         # 精调每步转 10°

    rospy.loginfo(f"[simple_turn_to] target={target_deg} deg, tol={tol_deg} deg")

    while not rospy.is_shutdown():
        # --- 超时检查 ---
        if rospy.Time.now() > deadline:
            rospy.logwarn(f"[simple_turn_to] timeout at {target_deg} deg")
            uav.stop()
            return False

        # --- 读取 yaw ---
        yaw = get_yaw(uav)
        if yaw is None:
            rospy.sleep(0.2)
            continue

        # --- 计算误差 ---
        err = target_deg - yaw
        err = normalize_180(err)

        rospy.loginfo(f"[simple_turn_to] yaw={yaw:.1f}  target={target_deg}  err={err:+.1f}  phase={phase}")

        # --- 到达判断 ---
        if abs(err) <= tol_deg:
            uav.stop()
            rospy.loginfo(f"[simple_turn_to] done: yaw={yaw:.1f} (err={err:.1f})")
            return True

        # --- 选择步长 ---
        if abs(err) <= 25:
            phase = "fine"
        else:
            phase = "coarse"

        if phase == "coarse":
            step = min(abs(err), coarse_step)
        else:
            step = min(abs(err), fine_step)

        # --- 发旋转指令 ---
        if err > 0:
            rospy.loginfo(f"  → ccw {int(step)}")
            uav.ccw(int(step))
        else:
            rospy.loginfo(f"  → cw {int(step)}")
            uav.cw(int(step))

        # 等待无人机执行：粗调等更久
        rospy.sleep(1.2 if phase == "coarse" else 0.8)


def main():
    rospy.init_node("test_turn_to_node", anonymous=False)
    name = rospy.get_param('~name', "")
    uav = TelloControl(name)

    # --- 起飞 ---
    rospy.loginfo("Taking off...")
    uav.takeoff()

    # 等待起飞完成
    init_z = 0
    for _ in range(10):
        with uav.state.lock:
            if uav.state.position is not None:
                init_z = uav.state.position.z
                break
        rospy.sleep(0.3)

    deadline = rospy.Time.now() + rospy.Duration(15)
    while not rospy.is_shutdown():
        with uav.state.lock:
            pz = uav.state.position.z if uav.state.position else init_z
        if pz - init_z > 0.3:
            rospy.loginfo(f"Takeoff OK (height={pz:.2f}m)")
            break
        if rospy.Time.now() > deadline:
            rospy.logerr("Takeoff timeout!")
            uav.land()
            return
        rospy.sleep(0.5)

    rospy.sleep(2)

    # --- 交互循环 ---
    print("\n" + "=" * 50)
    print("  输入目标角度（90 的整数倍），输入 q 退出")
    print("  例如: 0, 90, -90, 180, -180, 270, 360")
    print("=" * 50 + "\n")

    while not rospy.is_shutdown():
        try:
            raw = input("目标角度 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if raw.lower() in ("q", "quit", "exit"):
            break
        if not raw:
            continue

        try:
            target = int(raw)
        except ValueError:
            print(f"无效输入: '{raw}'，请输入整数角度值")
            continue

        if target % 90 != 0:
            print(f"注意: {target} 不是 90 的整数倍，继续执行...")

        ok = simple_turn_to(uav, target, tol_deg=10, timeout=20)
        if ok:
            print(f"→ 到位! (目标 {target}°)")
        else:
            print(f"→ 超时未到位 (目标 {target}°)")

    # --- 降落 ---
    rospy.loginfo("Landing...")
    uav.land()
    rospy.sleep(3)


if __name__ == "__main__":
    main()
