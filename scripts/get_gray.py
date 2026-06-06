#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像转灰度图工具。
用法：python get_gray.py <输入图片路径> [输出图片路径]
  不指定输出路径则保存为 <输入名>_gray.jpg
"""

import sys
import cv2
import os


def main():
    if len(sys.argv) < 2:
        print("用法: python get_gray.py <输入图片路径> [输出图片路径]")
        sys.exit(1)

    in_path = sys.argv[1]
    bgr = cv2.imread(in_path)
    if bgr is None:
        print(f"无法读取图片: {in_path}")
        sys.exit(1)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}_gray.jpg"

    cv2.imwrite(out_path, gray)
    print(f"灰度图已保存: {out_path}")


if __name__ == "__main__":
    main()
