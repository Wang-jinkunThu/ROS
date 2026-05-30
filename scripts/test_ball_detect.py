#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小球识别测试脚本。
用法:
  python test_ball_detect.py --image 照片.jpg     # 从 ./images/ 读取
  python test_ball_detect.py <图片路径>            # 直接指定路径
"""

import argparse
import os
import sys
import cv2
import numpy as np

TARGET_W = 640
TARGET_H = 480


def crop_to_4_3(img):
    """将图像中心裁剪为 4:3 比例，再缩放到 640x480"""
    h, w = img.shape[:2]
    target_ratio = 4.0 / 3.0

    if w / h > target_ratio:
        # 太宽，裁左右
        new_w = int(h * target_ratio)
        offset = (w - new_w) // 2
        cropped = img[:, offset:offset + new_w]
    else:
        # 太高，裁上下
        new_h = int(w / target_ratio)
        offset = (h - new_h) // 2
        cropped = img[offset:offset + new_h, :]

    return cv2.resize(cropped, (TARGET_W, TARGET_H))


def detect_ball_color(frame):
    """检测图像中球的颜色，返回 (result, green_area, orange_area, mask_green, mask_orange)。
    与 test_1.py 中 TelloControl.detect_ball_color 参数一致。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([30, 100, 80])
    upper_green = np.array([45, 255, 255])
    lower_orange = np.array([5, 80, 70])
    upper_orange = np.array([30, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel)

    green_area = cv2.countNonZero(mask_green)
    orange_area = cv2.countNonZero(mask_orange)

    if max(green_area, orange_area) < 200:
        return None, green_area, orange_area, mask_green, mask_orange
    if green_area > orange_area:
        return 'g', green_area, orange_area, mask_green, mask_orange
    else:
        return 'r', green_area, orange_area, mask_green, mask_orange


def make_result_image(frame, result, mask_green, mask_orange):
    """在剪裁图上绘制检测轮廓 + 顶端居中识别结果"""
    out = frame.copy()

    # 绿色轮廓（粗实线）
    contours_g, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours_g, -1, (0, 255, 0), 3)

    # 橙色轮廓（粗实线）
    contours_o, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours_o, -1, (0, 140, 255), 3)

    # 顶端居中文字
    color_name = {"g": "GREEN", "r": "ORANGE"}
    label = color_name.get(result, "NONE")
    text = f"Result: {label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cx = (frame.shape[1] - tw) // 2
    cv2.putText(out, text, (cx, th + 15), font, scale, (255, 255, 255), thickness + 2)
    cv2.putText(out, text, (cx, th + 15), font, scale, (0, 0, 0), thickness)

    return out


IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")


def list_images():
    """列出 ./images/ 下所有图片文件"""
    if not os.path.isdir(IMAGES_DIR):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    files = sorted(f for f in os.listdir(IMAGES_DIR)
                   if os.path.splitext(f)[-1].lower() in exts)
    return files


def main():
    parser = argparse.ArgumentParser(description="小球识别测试")
    parser.add_argument("--image", "-i", type=str, default=None,
                        help="图片文件名（位于 ./images/ 下）")
    parser.add_argument("path", nargs="?", type=str, default=None,
                        help="直接指定图片路径（兼容旧用法）")
    args = parser.parse_args()

    # 确定图片路径
    if args.image:
        path = os.path.join(IMAGES_DIR, args.image)
    elif args.path:
        path = args.path
    else:
        images = list_images()
        if images:
            print(f"可用的图片 ({IMAGES_DIR}):")
            for f in images:
                print(f"  {f}")
            print(f"\n用法: python test_ball_detect.py --image {images[0]}")
        else:
            print(f"未找到图片目录或目录为空: {IMAGES_DIR}")
            print("用法: python test_ball_detect.py --image 照片名.jpg")
        sys.exit(0)

    img = cv2.imread(path)
    if img is None:
        print(f"无法读取图片: {path}")
        sys.exit(1)

    # 裁剪 + 缩放
    processed = crop_to_4_3(img)

    # 保存裁剪图
    base, ext = os.path.splitext(path)
    cropped_path = f"{base}_cropped{ext}"
    cv2.imwrite(cropped_path, processed)
    print(f"[SAVE] 裁剪图: {cropped_path}")

    # 检测
    result, green_area, orange_area, mask_g, mask_o = detect_ball_color(processed)

    # 生成并保存结果图
    result_img = make_result_image(processed, result, mask_g, mask_o)
    result_path = f"{base}_detect_result{ext}"
    cv2.imwrite(result_path, result_img)
    print(f"[SAVE] 结果图: {result_path}")

    # 显示（标题为识别结果）
    color_name = {"g": "GREEN", "r": "ORANGE"}
    if result:
        title = f"Result: {color_name[result]}  (G={green_area}, O={orange_area})"
    else:
        title = f"Result: NONE  (G={green_area}, O={orange_area})"
    cv2.imshow(title, result_img)
    print(f"[DETECT] {title}")
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
