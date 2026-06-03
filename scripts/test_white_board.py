#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线测试脚本：输入手机拍的横向白板图片，检测白板轮廓和红色灯位置。
输出：白板 2D 坐标 + 三维世界坐标。
用法：python test_white_board.py <图片路径>
"""

import cv2
import numpy as np
import sys

# 白板 3×3 目标点 — 白板二维坐标系 (board_x, board_y)，原点左上
BOARD_2D_POINTS = [
    (25, 12.5), (100, 12.5), (175, 12.5),
    (25, 50),   (100, 50),   (175, 50),
    (25, 87.5), (100, 87.5), (175, 87.5),
]

# 白板 3×3 目标点 — 三维世界坐标 (world_x, world_z)，y=220
WORLD_3D_POINTS = [
    (-75, 150), (0, 150), (75, 150),
    (-75, 112.5), (0, 112.5), (75, 112.5),
    (-75, 75), (0, 75), (75, 75),
]


def detect_whiteboard_corners(gray):
    """检测白板区域，返回四个角点的像素坐标 [左上, 右上, 右下, 左下]"""
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    max_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)
    if len(approx) != 4:
        hull = cv2.convexHull(max_contour)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        if len(approx) != 4:
            return None
    pts = approx.reshape(4, 2)
    pts_sorted = sorted(pts, key=lambda p: (p[1], p[0]))
    top_pts = sorted(pts_sorted[:2], key=lambda p: p[0])          # [TL, TR]
    bottom_pts = sorted(pts_sorted[2:], key=lambda p: -p[0])      # [BR, BL]，x 降序
    corners = top_pts + bottom_pts                                 # [TL, TR, BR, BL]
    return np.array(corners, dtype='float32')


def detect_red_lights(bgr, board_corners=None):
    """检测红色光点，返回像素坐标列表 [(u, v), ...]。可传入 board_corners 过滤板外噪点。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 56, 244])
    upper_red1 = np.array([179, 124, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lights = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])
            # 若提供了白板角点，仅保留板内的红灯
            if board_corners is not None:
                inside = cv2.pointPolygonTest(board_corners, (float(u), float(v)), False)
                if inside < 0:
                    continue
            lights.append((u, v))
    return lights


def pixel_to_world(u, v, corners):
    """像素坐标 → 世界坐标 (x, z)，吸附到最近的 3×3 网格点"""
    world_pts = np.array([
        [-75, 150],  # 左上
        [75, 150],   # 右上
        [75, 75],    # 右下
        [-75, 75]    # 左下
    ], dtype='float32')

    H, _ = cv2.findHomography(corners, world_pts)
    pixel = np.array([u, v, 1], dtype='float32').reshape(3, 1)
    world = H @ pixel
    world = world / world[2]
    x = world[0, 0]
    z = world[1, 0]
    x = max(-75, min(75, x))
    z = max(75, min(150, z))

    best = min(WORLD_3D_POINTS, key=lambda p: (p[0] - x)**2 + (p[1] - z)**2)
    return best[0], best[1]


def world_to_board(x_world, z_world):
    """世界坐标 (x, z) → 白板二维坐标 (board_x, board_y)"""
    board_x = x_world + 100
    board_y = 162.5 - z_world
    return board_x, board_y


def cluster_lights(lights, pixel_dist=80):
    """将像素距离 < pixel_dist 的红灯聚为一组。返回最大簇的索引列表，若全孤立则返回空。"""
    n = len(lights)
    if n <= 1:
        return []

    # 邻接表：距离 < pixel_dist 视为相邻
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(lights[i][0] - lights[j][0], lights[i][1] - lights[j][1])
            if dist < pixel_dist:
                adj[i].append(j)
                adj[j].append(i)

    # DFS 找连通分量
    visited = [False] * n
    clusters = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                v = stack.pop()
                comp.append(v)
                for nb in adj[v]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            clusters.append(comp)

    # 优先返回点数最多的簇；若不存在 ≥2 的簇则返回空（全视为噪点）
    best = max(clusters, key=len)
    return best if len(best) >= 2 else []


def main():
    if len(sys.argv) < 2:
        print("用法: python test_white_board.py <图片路径>")
        sys.exit(1)

    img_path = sys.argv[1]
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"无法读取图片: {img_path}")
        sys.exit(1)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ========================
    #  1. 检测白板轮廓
    # ========================
    corners = detect_whiteboard_corners(gray)
    if corners is None:
        print("未检测到白板轮廓")
        cv2.imshow("Result", bgr)
        cv2.waitKey(0)
        return

    print("检测到白板轮廓")
    for i, label in enumerate(["左上", "右上", "右下", "左下"]):
        x, y = corners[i]
        print(f"  {label}: 像素 ({x:.0f}, {y:.0f})")

    # 画白板轮廓（白色线框）
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(bgr, [pts], True, (255, 255, 255), 3)
    labels = ["TL", "TR", "BR", "BL"]
    for (x, y), label in zip(corners.astype(int), labels):
        cv2.circle(bgr, (x, y), 8, (255, 255, 255), -1)
        cv2.putText(bgr, label, (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ========================
    #  2. 检测红色灯点 + 聚类
    # ========================
    all_lights = detect_red_lights(bgr, corners)
    cluster_idx = cluster_lights(all_lights)
    lights = [all_lights[i] for i in cluster_idx]
    is_clustered = len(cluster_idx) >= 2 and len(cluster_idx) < len(all_lights)

    if is_clustered:
        print(f"\n检测到 {len(all_lights)} 个红点，其中 {len(lights)} 个距离很近 → 输出这些点")
        print(f"  (忽略 {len(all_lights) - len(lights)} 个孤立红点)")
    else:
        print(f"\n检测到 {len(lights)} 个红色灯点")

    print("=" * 55)

    active_set = set(cluster_idx)
    for idx, (u, v) in enumerate(all_lights):
        x_w, z_w = pixel_to_world(u, v, corners)
        bx, by = world_to_board(x_w, z_w)
        grid_idx = WORLD_3D_POINTS.index((x_w, z_w))
        row = grid_idx // 3 + 1
        col = grid_idx % 3 + 1

        if idx in active_set:
            print(f"\n红灯 {idx + 1}:")
            print(f"  像素位置:        ({u}, {v})")
            print(f"  白板 2D 坐标:    ({bx:.1f}, {by:.1f}) cm")
            print(f"  世界 3D 坐标:    ({x_w:.1f}, 220, {z_w:.1f}) cm")
            print(f"  网格位置:        第 {row} 行, 第 {col} 列")

            # 活跃簇：红色圆圈
            cv2.circle(bgr, (u, v), 25, (0, 0, 255), 4)
            cv2.circle(bgr, (u, v), 5, (0, 0, 255), -1)
            cv2.putText(bgr, f"L{idx + 1}", (u + 30, v - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # 孤立噪点：灰色圆圈
            cv2.circle(bgr, (u, v), 20, (128, 128, 128), 2)
            cv2.putText(bgr, f"x", (u - 10, v + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)

    if not lights:
        print("  (未检测到红灯)")

    print("\n" + "=" * 55)
    print("按任意键关闭窗口...")

    # 缩放显示
    h, w = bgr.shape[:2]
    scale = min(1200 / max(w, 1), 800 / max(h, 1), 1.0)
    if scale < 1.0:
        display = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    else:
        display = bgr
    cv2.imshow("Whiteboard Detection | White=Board  Red=Light", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
