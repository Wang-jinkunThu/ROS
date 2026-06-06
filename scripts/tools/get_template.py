#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
截取模板工具：在图片上框选矩形区域，保存为模板。
用法：python get_template.py <输入图片路径>
输出：./template/TEMPLATE_<原名字>.png
"""

import sys
import os
import cv2


def main():
    if len(sys.argv) < 2:
        print("用法: python get_template.py <输入图片路径>")
        sys.exit(1)

    in_path = sys.argv[1]
    bgr = cv2.imread(in_path)
    if bgr is None:
        print(f"无法读取图片: {in_path}")
        sys.exit(1)

    roi = cv2.selectROI("框选模板区域 → 按 SPACE/ENTER 确认, C 取消", bgr, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("未选择区域，退出")
        sys.exit(0)

    template = bgr[y:y+h, x:x+w]

    save_dir = os.path.join(".", "template")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    basename = os.path.splitext(os.path.basename(in_path))[0]
    out_path = os.path.join(save_dir, f"TEMPLATE_{basename}.png")
    cv2.imwrite(out_path, template)
    print(f"模板已保存: {out_path}  (尺寸: {w}x{h})")


if __name__ == "__main__":
    main()
