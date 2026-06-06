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

TEMPLATE_PATH = "template/EMPTY.png"  # 替换为实际模板路径


def main():
    if len(sys.argv) < 2:
        print("用法: python test_find_template.py <图片路径>")
        sys.exit(1)

    img_path = sys.argv[1]
    tmpl_path = TEMPLATE_PATH

    bgr = cv2.imread(img_path)
    template = cv2.imread(tmpl_path)
    if bgr is None:
        print(f"无法读取图片: {img_path}")
        sys.exit(1)
    if template is None:
        print(f"无法读取模板: {tmpl_path}")
        sys.exit(1)

    th, tw = template.shape[:2]
    if th > bgr.shape[0] or tw > bgr.shape[1]:
        print("模板尺寸大于原图，无法匹配")
        sys.exit(1)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_tmpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray, gray_tmpl, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))  # (x, y) 列表

    # 非极大值抑制，避免重叠框
    min_dist = min(tw, th) // 2
    keep = []
    for loc in locations:
        if all(np.hypot(loc[0] - k[0], loc[1] - k[1]) > min_dist for k in keep):
            keep.append(loc)

    print(f"检测到 {len(keep)} 个模板匹配")

    for i, (x, y) in enumerate(keep):
        conf = result[y, x]
        print(f"  [{i+1}] 位置: ({x}, {y})  置信度: {conf:.4f}")
        cv2.rectangle(bgr, (x, y), (x + tw, y + th), (0, 0, 255), 3)
        cv2.putText(bgr, f"{conf:.3f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(os.path.dirname(img_path), f"{base}_template_match.jpg")
    cv2.imwrite(out_path, bgr)
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
