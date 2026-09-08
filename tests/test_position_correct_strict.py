# -*- coding: utf-8 -*-
"""位置修正严格自测:小角度旋转方向、大模板缩放路径、校正一致性、流水线联动。"""
import sys
sys.path.insert(0, r"D:\VScode Project\Python project\Vision_Substrate_Silicone")

import numpy as np
import cv2

from vision.pipeline import PipelineContext
from vision.tools.position import PositionCorrect, MAX_TEMPLATE_LONG_SIDE

FAIL = 0
PASS = 0


def check(name, cond, detail=""):
    global FAIL, PASS
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def make_ref_image(w=1400, h=1000):
    img = np.full((h, w, 3), (50, 60, 70), dtype=np.uint8)
    cv2.rectangle(img, (150, 100), (1250, 900), (150, 160, 170), -1)  # 板
    # 独特角标特征(L形),故意不对称
    cv2.rectangle(img, (180, 130), (330, 160), (20, 25, 35), -1)
    cv2.rectangle(img, (180, 130), (210, 300), (20, 25, 35), -1)
    # 右下角小方块(破坏对称性,帮助判断角度方向)
    cv2.rectangle(img, (1150, 800), (1210, 860), (20, 25, 35), -1)
    # 中部图案
    cv2.circle(img, (700, 500), 90, (200, 70, 60), -1)
    cv2.rectangle(img, (600, 700), (760, 800), (40, 200, 90), -1)
    return img


def build_tool(ref_img, box):
    x, y, w, h = box
    t = PositionCorrect()
    t.set_template(ref_img[y:y + h, x:x + w].copy(), x, y)
    return t


def run(tool, img):
    ctx = PipelineContext(original_image=img, current_image=img)
    return tool.process(ctx)


def test_positive_angle():
    """参考图逆时针旋转 +8°,校正后应回到参考姿态。"""
    print("\n[1] +8° 小角度旋转校正(逆时针)")
    ref = make_ref_image()
    tool = build_tool(ref, (180, 130, 150, 170))
    tool.params["rot_mode"] = "range"
    tool.params["angle_min"], tool.params["angle_max"] = -15, 15
    tool.params["angle_step"] = 2
    h, w = ref.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.0)
    rotated = cv2.warpAffine(ref, M, (w, h), borderValue=(50, 60, 70))
    res = run(tool, rotated)
    check("matched(+8°)", bool(res.passed), f"score={res.data.get('score')}")
    ang = res.data.get("angle_deg")
    check("检出角度≈+8", bool(res.passed) and abs(ang - 8) <= 3,
          f"angle={ang}")
    if res.processed_image is not None:
        # 只比较板身内部区域(避开黑边)
        crop_ref = ref[200:800, 300:1200]
        crop_cor = res.processed_image[200:800, 300:1200]
        mean = cv2.absdiff(crop_ref, crop_cor).mean()
        check("校正后内部≈参考", mean < 20, f"mean_diff={mean:.2f}")


def test_negative_angle():
    """-6° 方向验证(顺时针),防止方向搞反。"""
    print("\n[2] -6° 小角度(顺时针)")
    ref = make_ref_image()
    tool = build_tool(ref, (180, 130, 150, 170))
    tool.params["rot_mode"] = "range"
    tool.params["angle_min"], tool.params["angle_max"] = -15, 15
    tool.params["angle_step"] = 2
    h, w = ref.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -6, 1.0)
    rotated = cv2.warpAffine(ref, M, (w, h), borderValue=(50, 60, 70))
    res = run(tool, rotated)
    ang = res.data.get("angle_deg")
    check("matched(-6°)", bool(res.passed), f"score={res.data.get('score')}")
    check("检出角度≈-6", bool(res.passed) and abs(ang - (-6)) <= 3,
          f"angle={ang}")
    if res.processed_image is not None:
        crop_ref = ref[200:800, 300:1200]
        crop_cor = res.processed_image[200:800, 300:1200]
        mean = cv2.absdiff(crop_ref, crop_cor).mean()
        check("校正后内部≈参考", mean < 20, f"mean_diff={mean:.2f}")


def test_large_template_scale():
    """超大模板触发缩小路径:验证坐标换算正确性。

    注:大模板含较多均匀板底,TM_CCOEFF 分数天然偏低,属"基准框选太大"的
    现场配置问题而非逻辑错误;此处把阈值放宽以专门验证几何换算。
    """
    print("\n[3] 大模板缩放路径(>600px 上限,几何正确性)")
    ref = make_ref_image()
    # 框一个很大的模板(触发 MAX_TEMPLATE_LONG_SIDE 缩放)
    box = (180, 130, 700, 500)   # 长边 ~700 > 600
    tool = build_tool(ref, box)
    tool.params["rot_mode"] = "0"
    tool.params["threshold"] = 0.4   # 大均匀模板分数偏低,仅测几何
    # 平移 60,40
    M = np.float32([[1, 0, 60], [0, 1, 40]])
    moved = cv2.warpAffine(ref, M, (ref.shape[1], ref.shape[0]),
                           borderValue=(50, 60, 70))
    res = run(tool, moved)
    check("matched(大模板)", bool(res.passed), f"score={res.data.get('score')}")
    if res.passed and res.processed_image is not None:
        crop_ref = ref[300:700, 400:1100]
        crop_cor = res.processed_image[300:700, 400:1100]
        mean = cv2.absdiff(crop_ref, crop_cor).mean()
        check("大模板校正后内部≈参考", mean < 20, f"mean_diff={mean:.2f}")


def test_shift_pixel_exact():
    """精确像素平移校正。"""
    print("\n[4] 像素级平移校正")
    ref = make_ref_image()
    tool = build_tool(ref, (180, 130, 150, 170))
    tool.params["rot_mode"] = "0"
    M = np.float32([[1, 0, 55], [0, 1, -35]])
    moved = cv2.warpAffine(ref, M, (ref.shape[1], ref.shape[0]),
                           borderValue=(50, 60, 70))
    res = run(tool, moved)
    d = res.data
    check("dx≈55", d.get("matched") and abs(d["dx"] - 55) <= 1.5,
          f"dx={d.get('dx')}")
    check("dy≈-35", d.get("matched") and abs(d["dy"] - (-35)) <= 1.5,
          f"dy={d.get('dy')}")
    if res.passed and res.processed_image is not None:
        crop_ref = ref[300:700, 400:1100]
        crop_cor = res.processed_image[300:700, 400:1100]
        mean = cv2.absdiff(crop_ref, crop_cor).mean()
        check("平移校正内部≈参考", mean < 15, f"mean_diff={mean:.2f}")


def test_flip_pipeline():
    """180° 翻转 + 胶垫颜色检测联动(整条链路)。胶垫避开图像翻转对称中心。"""
    print("\n[5] 翻转场景流水线联动")
    ref = make_ref_image()
    pad_box = (900, 650, 160, 160)   # 不在翻转对称中心 → 对照组必 NG
    hsv = np.array([[[30, 180, 200]]], np.uint8)
    bgr = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
    cv2.rectangle(ref, (pad_box[0] + 30, pad_box[1] + 30),
                  (pad_box[0] + pad_box[2] - 30, pad_box[1] + pad_box[3] - 30),
                  bgr, -1)
    tool_pos = build_tool(ref, (180, 130, 150, 170))
    tool_pos.params["rot_mode"] = "0and180"
    from vision.pipeline import Pipeline, create_tool
    pipe = Pipeline("联动")
    pipe.add_step(tool_pos)
    pipe.add_step(create_tool("MultiROI", {
        "regions": [{"name": "PAD", "x": pad_box[0], "y": pad_box[1],
                     "width": pad_box[2], "height": pad_box[3], "enabled": True}],
        "use_percentage": False}))
    color = create_tool("ColorRecognition", {
        "color_model": {"name": "胶", "color_space": "HSV", "center": [30, 180, 200],
                        "match_mode": "range", "tolerance": [10, 60, 60],
                        "distance_threshold": 40.0, "cluster_centers": [],
                        "normalize_illumination": False, "adaptive_threshold": False,
                        "source": "custom"},
        "match_mode": "range", "distance_threshold": 40.0,
        "normalize_illumination": False, "adaptive_threshold": False,
        "pass_min": 20.0, "pass_max": 90.0, "_input_source": "region:PAD",
    })
    pipe.add_step(color)

    flipped = cv2.rotate(ref, cv2.ROTATE_180)
    ok, results, _, _ = pipe.execute(flipped)
    check("翻转后整板仍OK(位置修正把胶垫转回参考位)", bool(ok),
          f"failed={[r.message for r in results if not bool(r.passed)]}")
    # 对照组:无位置修正
    pipe2 = Pipeline("对照")
    pipe2.add_step(create_tool("MultiROI", {
        "regions": [{"name": "PAD", "x": pad_box[0], "y": pad_box[1],
                     "width": pad_box[2], "height": pad_box[3], "enabled": True}],
        "use_percentage": False}))
    pipe2.add_step(color)
    ok2, results2, _, _ = pipe2.execute(flipped)
    check("对照组(翻转无修正)NG", bool(ok2) is False,
          f"意外通过: {[r.message for r in results2 if bool(r.passed)]}")


if __name__ == "__main__":
    test_positive_angle()
    test_negative_angle()
    test_large_template_scale()
    test_shift_pixel_exact()
    test_flip_pipeline()
    print(f"\n结果: PASS={PASS} FAIL={FAIL}")
    sys.exit(1 if FAIL else 0)
