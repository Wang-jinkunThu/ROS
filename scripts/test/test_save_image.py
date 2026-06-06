#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拍照测试脚本：不控制无人机移动，仅通过键盘拍照。
  y → 拍照，保存为 img_<时间戳>.jpg
  n → 退出
用法：python test_save_image.py <无人机名称>
"""

import sys
import os
import time
import threading
import cv2
import rospy
from Tello import TelloControl
  

class PhotoCapture:
    def __init__(self, drone_name):
        self.uav = TelloControl(drone_name)
        self.save_dir = os.path.join(os.path.expanduser("~"), "saved_images")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        rospy.loginfo(f"照片保存目录: {self.save_dir}")

    def capture(self):
        """拍照并保存"""
        with self.uav.state.lock:
            if self.uav.state.image is None:
                rospy.logwarn("暂无图像，请稍后再试")
                return None
            frame = self.uav.state.image.copy()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"img_{timestamp}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, frame)
        rospy.loginfo(f"已保存: {filepath}")
        return filepath

    def run(self):
        rospy.loginfo("拍照就绪 — 输入 y 拍照, n 退出")
        while not rospy.is_shutdown():
            try:
                key = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if key == 'y':
                self.capture()
            elif key == 'n':
                rospy.loginfo("退出")
                break
            else:
                print("  y = 拍照  |  n = 退出")


if __name__ == "__main__":
    rospy.init_node("test_save_image_node", anonymous=False)
    name = rospy.get_param('~name', '')
    if not name:
        print("用法: python test_save_image.py <无人机名称>")
        print("或: rosrun <pkg> test_save_image.py _name:=<无人机名称>")
        sys.exit(1)

    capturer = PhotoCapture(name)
    capturer.run()
