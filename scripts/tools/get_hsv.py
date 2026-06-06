#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSV 范围分析工具。
用法: python get_hsv.py <图片路径>
操作: 鼠标左键依次点击多边形顶点 → 右键闭合 → 按任意键关闭
"""

import argparse
import os
import sys
import cv2
import numpy as np

IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")
POINTS = []          # 用户点击的多边形顶点
IMG = None           # 当前显示的图像（用于鼠标回调）
IMG_HSV = None       # HSV 图像


def on_mouse(event, x, y, flags, param):
    """鼠标回调：左键加点，右键闭合"""
    if event == cv2.EVENT_LBUTTONDOWN:
        POINTS.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and len(POINTS) >= 3:
        POINTS.append(POINTS[0])  # 闭合多边形


def draw_polygon(img):
    """在图像上绘制已选多边形"""
    vis = img.copy()
    for i, pt in enumerate(POINTS):
        cv2.circle(vis, pt, 3, (0, 255, 0), -1)
        if i > 0:
            cv2.line(vis, POINTS[i - 1], POINTS[i], (0, 255, 0), 2)
    return vis


def compute_stats(hsv, mask):
    """计算 mask 区域内 HSV 的均值、标准差、范围"""
    pixels = hsv[mask > 0]
    if len(pixels) == 0:
        return None
    mean = np.mean(pixels, axis=0)
    std = np.std(pixels, axis=0)
    vmin = np.min(pixels, axis=0)
    vmax = np.max(pixels, axis=0)
    return {"mean": mean, "std": std, "min": vmin, "max": vmax, "count": len(pixels)}


def recommend_bounds(stats, n_sigma=2.0):
    """根据均值±n倍标准差推荐 HSV 上下界"""
    lower = stats["mean"] - n_sigma * stats["std"]
    upper = stats["mean"] + n_sigma * stats["std"]
    lower = np.clip(lower, 0, [179, 255, 255])
    upper = np.clip(upper, 0, [179, 255, 255])
    lower = lower.astype(int)
    upper = upper.astype(int)
    return lower, upper


def print_stats(label, stats):
    """格式化打印统计信息"""
    if stats is None:
        print(f"\n{'='*60}")
        print(f"  {label}: 无像素")
        return None
    m, s = stats["mean"], stats["std"]
    print(f"\n{'='*60}")
    print(f"  {label}  (像素数: {stats['count']})")
    print(f"    H 均值={m[0]:.1f}  标准差={s[0]:.1f}  范围=[{stats['min'][0]:.0f}, {stats['max'][0]:.0f}]")
    print(f"    S 均值={m[1]:.1f}  标准差={s[1]:.1f}  范围=[{stats['min'][1]:.0f}, {stats['max'][1]:.0f}]")
    print(f"    V 均值={m[2]:.1f}  标准差={s[2]:.1f}  范围=[{stats['min'][2]:.0f}, {stats['max'][2]:.0f}]")

    lower, upper = recommend_bounds(stats)
    print(f"    推荐 lower = np.array([{lower[0]}, {lower[1]}, {lower[2]}])")
    print(f"    推荐 upper = np.array([{upper[0]}, {upper[1]}, {upper[2]}])")
    return lower, upper


def main():
    global IMG, IMG_HSV

    parser = argparse.ArgumentParser(description="HSV 范围分析工具")
    parser.add_argument("--image", "-i", type=str, default=None,
                        help="图片文件名（位于 ./images/ 下）")
    parser.add_argument("path", nargs="?", type=str, default=None,
                        help="直接指定图片路径")
    args = parser.parse_args()

    if args.image:
        path = os.path.join(IMAGES_DIR, args.image)
    elif args.path:
        path = args.path
    else:
        print("用法: python get_hsv.py --image 照片名.jpg")
        sys.exit(0)

    raw = cv2.imread(path)
    if raw is None:
        print(f"无法读取图片: {path}")
        sys.exit(1)

    # 显示尺寸限制（过大则缩小便于操作）
    h, w = raw.shape[:2]
    max_display = 900
    scale = 1.0
    if max(h, w) > max_display:
        scale = max_display / max(h, w)
        raw = cv2.resize(raw, (int(w * scale), int(h * scale)))

    IMG = raw.copy()
    IMG_HSV = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)

    cv2.namedWindow("Select Region")
    cv2.setMouseCallback("Select Region", on_mouse)

    print("操作说明:")
    print("  鼠标左键: 依次点击多边形顶点")
    print("  鼠标右键: 闭合多边形并输出结果")
    print("  按 'r'  : 重置选点")
    print("  按 ESC  : 退出")
    print("")

    while True:
        vis = draw_polygon(IMG)
        cv2.imshow("Select Region", vis)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            print("退出。")
            cv2.destroyAllWindows()
            return
        elif key == ord('r'):
            POINTS.clear()
            print("已重置选点。")

        # 检测右键闭合（POINTS 末尾被设为起始点）
        if len(POINTS) >= 4 and POINTS[-1] == POINTS[0]:
            # 构建前景 / 背景 mask
            pts_array = np.array(POINTS[:-1], dtype=np.int32)
            mask_fg = np.zeros(raw.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask_fg, [pts_array], 255)
            mask_bg = cv2.bitwise_not(mask_fg)

            # 计算统计
            fg_stats = compute_stats(IMG_HSV, mask_fg)
            bg_stats = compute_stats(IMG_HSV, mask_bg)

            # 输出
            print_stats("前景(选中区域)", fg_stats)
            print_stats("背景(其余区域)", bg_stats)

            # 显示分割结果
            segmented = IMG.copy()
            segmented[mask_fg > 0] = cv2.addWeighted(
                segmented[mask_fg > 0], 0.4,
                np.full_like(segmented[mask_fg > 0], (0, 255, 0)), 0.6, 0)
            cv2.imshow("Segmentation", segmented)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return


if __name__ == "__main__":
    main()
