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
    "LIGHT_ON":  ("template/LIGHT_ON.png",  (0, 255, 0)),      # 绿
    "LIGHT_OFF": (["template/LIGHT_OFF_1.png","template/LIGHT_OFF_2.png","template/LIGHT_OFF_3.png"], (0, 0, 255)),      # 红
}

# 若一个类型有多张样张，覆盖掉上面同名 key 即可（用列表替代单路径）
# 示例：LIGHT_ON 有 3 种样张，EMPTY 有 2 种
# TEMPLATES = {
#     "EMPTY":     (["template/EMPTY_1.png", "template/EMPTY_2.png"], (255, 0, 0)),
#     "LIGHT_ON":  (["template/LIGHT_ON_a.png", "template/LIGHT_ON_b.png", "template/LIGHT_ON_c.png"], (0, 255, 0)),
#     "LIGHT_OFF": ("template/LIGHT_OFF.png", (0, 0, 255)),
# }

THRESHOLD = 0.8


def match_template(gray_img, gray_tmpl):
    """匹配单个模板，返回原始匹配列表 [(x, y, confidence, tw, th), ...]"""
    th, tw = gray_tmpl.shape
    result = cv2.matchTemplate(gray_img, gray_tmpl, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(result >= THRESHOLD)
    return [(x, y, result[y, x], tw, th) for x, y in zip(xs, ys)]


def nms(matches, min_dist=None):
    """非极大值抑制，返回去重后的匹配列表"""
    if not matches:
        return []
    if min_dist is None:
        min_dist = min(min(m[3], m[4]) for m in matches) // 2
    matches = sorted(matches, key=lambda m: -m[2])  # 按置信度降序
    keep = []
    for m in matches:
        if all(np.hypot(m[0] - k[0], m[1] - k[1]) > min_dist for k in keep):
            keep.append(m)
    return keep


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
    total = 0

    for name, (paths, color) in TEMPLATES.items():
        if isinstance(paths, str):
            paths = [paths]

        all_matches = []  # 收集该类型所有样张的匹配
        for tmpl_path in paths:
            template = cv2.imread(tmpl_path)
            if template is None:
                print(f"无法读取模板: {tmpl_path}，跳过")
                continue

            gray_tmpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            raw = match_template(gray, gray_tmpl)
            print(f"[{name}] 模板 {tmpl_path}: {len(raw)} 个原始匹配")
            all_matches.extend(raw)

        # 合并后统一 NMS，避免同位置多框
        matches = nms(all_matches)
        print(f"[{name}] NMS 后保留 {len(matches)} 个匹配")

        for i, (x, y, conf, tw, th) in enumerate(matches):
            print(f"    [{i + 1}] 位置: ({x}, {y})  置信度: {conf:.4f}")
            cv2.rectangle(bgr, (x, y), (x + tw, y + th), color, 3)
            cv2.putText(bgr, f"{name} {conf:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        total += len(matches)

    print(f"\n共检测到 {total} 个目标")

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(os.path.dirname(img_path), f"{base}_template_match.jpg")
    cv2.imwrite(out_path, bgr)
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
