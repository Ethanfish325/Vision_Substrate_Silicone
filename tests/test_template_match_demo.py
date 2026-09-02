# -*- coding: utf-8 -*-
"""
模板匹配测试 Demo（多尺度匹配诊断）
=====================================
获取当前相机图像，按 PCBA.json 中的"label2"ROI 裁剪，
对模板 1.jpg 做多尺度缩放匹配，诊断是否因缩放导致匹配不到。

用法:
    python test_template_match_demo.py

功能:
    1. 连接相机并拍照（或从本地图像读取）。
    2. 加载模板 C:/Users/fyx/Desktop/1.jpg（尺寸 292x282）。
    3. 按 PCBA.json 的 label2 ROI 裁剪图像。
    4. 对模板做多尺度缩放（0.5x ~ 2.0x），在每个尺度下用
       TM_CCOEFF_NORMED 匹配，找出最佳缩放比例和分数。
    5. 同时测试 SIFT 特征点匹配。
    6. 打印结果并保存标注图。
"""
import os
import sys
import json
import cv2
import numpy as np

# 模板路径（与 PCBA.json 方案一致）
TEMPLATE_PATH = "C:/Users/fyx/Desktop/2026-08-31 085456.jpg"
# 若相机不可用，可指定本地图像测试（设为 None 则用相机）
IMAGE_PATH = None  # 例如 "C:/Users/fyx/Desktop/current.jpg"

# 方案文件（用于读取 label2 ROI）
SCHEME_PATH = "data/schemes/PCBA.json"

# 多尺度匹配的缩放范围
SCALE_MIN = 0.5
SCALE_MAX = 2.0
SCALE_STEP = 0.1

# 匹配阈值
CCOEFF_THRESHOLD = 0.5
MIN_MATCHES = 10


def get_camera_image():
    """从相机获取当前图像。"""
    from camera_manager import CameraManager, raw_to_opencv

    cam = CameraManager()
    try:
        devices = cam.enumerate_devices()
        if not devices:
            print("未找到相机设备")
            return None
        print(f"找到 {len(devices)} 个相机设备")
        cam.open_camera(devices[0])
    except Exception as e:
        print(f"打开相机失败: {e}")
        return None

    try:
        raw = cam.capture_once()
        if isinstance(raw, tuple) and len(raw) == 4:
            width, height, pixel_type, frame_data = raw
            img = raw_to_opencv(frame_data, width, height, pixel_type)
        else:
            img = raw
    except Exception as e:
        print(f"拍照失败: {e}")
        img = None

    cam.close_camera()
    return img


def load_roi(region_name="label2"):
    """从 PCBA.json 读取指定 ROI，返回 (x, y, w, h) 或 None。"""
    try:
        with open(SCHEME_PATH, "r", encoding="utf-8") as f:
            scheme = json.load(f)
        for step in scheme.get("steps", []):
            if step.get("tool_type") == "MultiROI":
                for region in step.get("params", {}).get("regions", []):
                    if region.get("name") == region_name:
                        return (int(region["x"]), int(region["y"]),
                                int(region["width"]), int(region["height"]))
    except Exception as e:
        print(f"读取方案 ROI 失败: {e}")
    return None


def multi_scale_match(img_gray, templ_gray, scale_min, scale_max, scale_step):
    """多尺度模板匹配，返回 (best_score, best_loc, best_scale, best_tw, best_th, results)。"""
    img_h, img_w = img_gray.shape[:2]
    th0, tw0 = templ_gray.shape[:2]
    best_score = -1.0
    best_loc = (0, 0)
    best_scale = 1.0
    best_tw, best_th = tw0, th0
    results = []

    scale = scale_min
    while scale <= scale_max + 1e-6:
        tw = max(1, int(round(tw0 * scale)))
        th = max(1, int(round(th0 * scale)))
        # 模板不能大于输入图像
        if tw > img_w or th > img_h:
            scale += scale_step
            continue
        scaled = cv2.resize(templ_gray, (tw, th), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(img_gray, scaled, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        results.append({"scale": scale, "tw": tw, "th": th,
                        "score": float(max_val), "loc": max_loc})
        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_scale = scale
            best_tw, best_th = tw, th
        scale += scale_step

    return best_score, best_loc, best_scale, best_tw, best_th, results


def match_sift(img_gray, templ_gray):
    """SIFT 特征点匹配，返回 (match_count, display)。"""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(templ_gray, None)
    kp2, des2 = sift.detectAndCompute(img_gray, None)
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return 0, None
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
    display = cv2.drawMatches(templ_gray, kp1, img_gray, kp2, good, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good), display


def main():
    print("=" * 60)
    print("模板匹配测试 Demo（多尺度匹配诊断）")
    print("=" * 60)

    # 1. 获取图像
    if IMAGE_PATH and os.path.exists(IMAGE_PATH):
        print(f"使用本地图像: {IMAGE_PATH}")
        img = cv2.imread(IMAGE_PATH)
    else:
        print("正在从相机获取图像...")
        img = get_camera_image()

    if img is None:
        print("无法获取图像！请确认相机已连接，或设置 IMAGE_PATH 使用本地图像。")
        return
    print(f"图像尺寸: {img.shape[1]}x{img.shape[0]}")

    # 2. 加载模板
    if not os.path.exists(TEMPLATE_PATH):
        print(f"模板文件不存在: {TEMPLATE_PATH}")
        return
    template = cv2.imread(TEMPLATE_PATH)
    if template is None:
        print(f"模板加载失败: {TEMPLATE_PATH}")
        return
    print(f"模板尺寸: {template.shape[1]}x{template.shape[0]}")

    # 3. 读取 label2 ROI 并裁剪
    roi = load_roi("label2")
    if roi is None:
        print("未找到 label2 ROI，将使用全图匹配")
        roi_img = img
        roi_rect = (0, 0, img.shape[1], img.shape[0])
    else:
        x, y, w, h = roi
        x = max(0, min(x, img.shape[1] - 1))
        y = max(0, min(y, img.shape[0] - 1))
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        roi_img = img[y:y + h, x:x + w]
        roi_rect = (x, y, w, h)
        print(f"label2 ROI: x={x}, y={y}, w={w}, h={h}")
        print(f"ROI 图像尺寸: {roi_img.shape[1]}x{roi_img.shape[0]}")

    # 4. 转灰度
    img_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    templ_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # 5. 多尺度匹配
    print("-" * 60)
    print("多尺度模板匹配（TM_CCOEFF_NORMED）")
    print(f"缩放范围: {SCALE_MIN}x ~ {SCALE_MAX}x，步长 {SCALE_STEP}")
    best_score, best_loc, best_scale, best_tw, best_th, results = multi_scale_match(
        img_gray, templ_gray, SCALE_MIN, SCALE_MAX, SCALE_STEP)

    # 打印所有尺度的分数
    print(f"{'缩放':>6} {'尺寸':>10} {'分数':>8} {'位置':>12}")
    for r in results:
        mark = " ← 最佳" if abs(r["scale"] - best_scale) < 1e-6 else ""
        print(f"{r['scale']:>6.1f} {r['tw']}x{r['th']:>5} {r['score']:>8.4f} {str(r['loc']):>12}{mark}")

    print("-" * 60)
    print(f"最佳缩放: {best_scale:.1f}x (模板 {best_tw}x{best_th})")
    print(f"最佳分数: {best_score:.4f} (位置 {best_loc})")
    print(f"阈值: {CCOEFF_THRESHOLD} → {'✅ 匹配' if best_score >= CCOEFF_THRESHOLD else '❌ 不匹配'}")

    # 6. SIFT 特征点匹配
    print("-" * 60)
    sift_count, disp_sift = match_sift(img_gray, templ_gray)
    print(f"SIFT 特征点匹配: {sift_count} 个匹配点 "
          f"{'✅' if sift_count >= MIN_MATCHES else '❌'}")

    # 7. 绘制标注
    annotated_roi = roi_img.copy()
    bx, by = best_loc
    cv2.rectangle(annotated_roi, (bx, by), (bx + best_tw, by + best_th), (0, 255, 0), 3)
    cv2.putText(annotated_roi, f"scale={best_scale:.1f} score={best_score:.3f}",
                (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    annotated_full = img.copy()
    rx, ry, rw, rh = roi_rect
    cv2.rectangle(annotated_full, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
    cv2.putText(annotated_full, "label2", (rx, ry - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    out_path = "template_match_result.jpg"
    cv2.imwrite(out_path, annotated_full)
    print(f"标注结果已保存: {out_path}")

    # 8. 显示图像
    try:
        def _resize_for_display(disp_img, max_w=900):
            h, w = disp_img.shape[:2]
            if w > max_w:
                s = max_w / w
                disp_img = cv2.resize(disp_img, (int(w * s), int(h * s)),
                                      interpolation=cv2.INTER_AREA)
            return disp_img

        cv2.imshow("相机图像(含ROI)", _resize_for_display(annotated_full))
        cv2.imshow("模板", _resize_for_display(template, 400))
        cv2.imshow("ROI匹配结果", _resize_for_display(annotated_roi))
        if disp_sift is not None:
            cv2.imshow("SIFT特征匹配", _resize_for_display(disp_sift))
        print("图像已显示，按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"显示图像失败: {e}")


if __name__ == "__main__":
    main()
