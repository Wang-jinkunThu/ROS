#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模板匹配测试：在输入图片中查找模板，框出匹配位置并保存结果。
用法：python test_find_template.py <图片路径>
"""

import sys
import os
import cv2
import numpy as np

TEMPLATES = {
    "EMPTY":     (["template/EMPTY_1.png", "template/EMPTY_2.png"],     (255, 0, 0)),      # 蓝
    "LIGHT_ON":  (["template/LIGHT_ON_1.png", "template/LIGHT_ON_2.png"],  (0, 255, 0)),      # 绿
    "LIGHT_OFF": (["template/LIGHT_OFF_1.png","template/LIGHT_OFF_2.png","template/LIGHT_OFF_3.png","template/LIGHT_OFF_4.png"], (0, 0, 255)),      # 红
}

THRESHOLD = 0.75

# 白板 3×3 目标点的三维世界坐标 (x, y, z)，行优先
WORLD_TARGETS = [
    (-75, 220, 150), (0, 220, 150), (75, 220, 150),
    (-75, 220, 112.5), (0, 220, 112.5), (75, 220, 112.5),
    (-75, 220, 75), (0, 220, 75), (75, 220, 75),
]


def main():
    if len(sys.argv) < 2:
        print("用法: python test_find_template.py <图片路径>")
        sys.exit(1)

    img_path = sys.argv[1]
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"无法读取图片: {img_path}")
        sys.exit(1)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ========================
    #  跨类型模板匹配 + 网格映射
    # ========================
    all_matches = []
    for name, (paths, _) in TEMPLATES.items():
        if isinstance(paths, str):
            paths = [paths]
        for tmpl_path in paths:
            template = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"无法读取模板: {tmpl_path}，跳过")
                continue
            th, tw = template.shape
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(result >= THRESHOLD)
            print(f"[{name}] 模板 {tmpl_path}: {len(xs)} 个原始匹配")
            for x, y in zip(xs, ys):
                all_matches.append((x, y, float(result[y, x]), name, tw, th))

    if not all_matches:
        print("未检测到任何匹配")
        return

    # 跨类型 NMS
    all_matches.sort(key=lambda m: -m[2])
    min_dist = min(min(m[4], m[5]) for m in all_matches) // 2
    keep = []
    for m in all_matches:
        if all(np.hypot(m[0] - k[0], m[1] - k[1]) > min_dist for k in keep):
            keep.append(m)
    print(f"\n跨类型 NMS 后保留 {len(keep)} 个点\n")

    # 3×3 网格映射
    if len(keep) < 3:
        print("检测点太少，无法构建 3×3 网格")
        return

    xs = [m[0] for m in keep]
    ys = [m[1] for m in keep]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cell_w = (x_max - x_min) / 3.0
    cell_h = (y_max - y_min) / 3.0

    # 分配点到网格单元
    grid_map = {}
    for m in keep:
        col = int((m[0] - x_min) / cell_w)
        row = int((m[1] - y_min) / cell_h)
        col = max(0, min(2, col))
        row = max(0, min(2, row))
        key = (row, col)
        if key not in grid_map or m[2] > grid_map[key][2]:
            grid_map[key] = m

    print("=" * 60)
    print(f"网格: bounding box ({x_min}, {y_min}) ~ ({x_max}, {y_max})")
    print(f"单元格: {cell_w:.0f} × {cell_h:.0f} px")
    print(f"映射到 {len(grid_map)} 个格子\n")

    colors = {k: v[1] for k, v in TEMPLATES.items()}
    for row in range(3):
        for col in range(3):
            key = (row, col)
            idx = row * 3 + col
            wx, wy, wz = WORLD_TARGETS[idx]
            if key in grid_map:
                m = grid_map[key]
                x, y, conf, ttype, tw, th = m
                color = colors[ttype]
                print(f"  ({row},{col}) {ttype:9s}  conf={conf:.3f}  pixel=({x},{y})  →  world ({wx}, {wy}, {wz})")

                cv2.rectangle(bgr, (x, y), (x + tw, y + th), color, 3)
                label = f"{ttype} {conf:.2f}"
                cv2.putText(bgr, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                print(f"  ({row},{col}) (空)                                  →  world ({wx}, {wy}, {wz})")
    print("=" * 60)

    # 高亮 LIGHT_ON
    light_on_found = False
    for (row, col), m in grid_map.items():
        if m[3] == "LIGHT_ON":
            idx = row * 3 + col
            wx, wy, wz = WORLD_TARGETS[idx]
            print(f"\n*** LIGHT_ON 世界坐标: ({wx}, {wy}, {wz}) ***")
            light_on_found = True
    if not light_on_found:
        print("\n未检测到 LIGHT_ON")

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(os.path.dirname(img_path), f"{base}_template_match.jpg")
    cv2.imwrite(out_path, bgr)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
