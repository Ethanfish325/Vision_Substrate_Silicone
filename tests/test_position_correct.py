# -*- coding: utf-8 -*-
"""位置修正算子行为自测(合成图像):偏移+180°翻转后校正应恢复。"""
import sys
sys.path.insert(0, r"D:\VScode Project\Python project\Vision_Substrate_Silicone")

import numpy as np
import cv2

from vision.pipeline import PipelineContext
from vision.tools.position import PositionCorrect

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    print(("  [PASS] " if cond else "  [FAIL] ") + name + ("  " + detail if detail and not cond else ""))
    if not cond:
        FAIL += 1


def make_ref_image(w=1200, h=900):
    """画一块'板子'：底色 + 一个独特角标特征(当作定位基准)。"""
    img = np.full((h, w, 3), (60, 70, 80), dtype=np.uint8)
    # 板身
    cv2.rectangle(img, (200, 150), (1000, 750), (150, 160, 170), -1)
    # 稳定特征：左上角 L 形角标(独特的图案便于模板匹配)
    cv2.rectangle(img, (240, 190), (420, 230), (30, 30, 40), -1)
    cv2.rectangle(img, (240, 190), (280, 380), (30, 30, 40), -1)
    # 其它装饰(避免全图过于空旷影响匹配)
    cv2.circle(img, (600, 450), 60, (200, 60, 60), -1)
    cv2.rectangle(img, (750, 500), (900, 650), (40, 200, 90), -1)
    return img


def build_tool(ref_img, feat_box):
    """从参考图裁特征生成工具。"""
    x, y, w, h = feat_box
    tool = PositionCorrect()
    tool.set_template(ref_img[y:y + h, x:x + w].copy(), x, y)
    return tool


def test_shift():
    print("\n[1] 仅平移校正")
    ref = make_ref_image()
    tool = build_tool(ref, (240, 190, 180, 190))  # L角标整体
    # 向右下平移 80,60
    M = np.float32([[1, 0, 80], [0, 1, 60]])
    moved = cv2.warpAffine(ref, M, (ref.shape[1], ref.shape[0]),
                           borderValue=(60, 70, 80))
    ctx = PipelineContext(original_image=moved, current_image=moved)
    res = tool.process(ctx)
    check("matched", bool(res.passed), f"score={res.data.get('score')}")
    d = res.data
    check("dx≈+80(特征右移)", d.get("matched") and abs(d["dx"] - 80) < 3, f"dx={d.get('dx')}")
    check("dy≈+60(特征下移)", d.get("matched") and abs(d["dy"] - 60) < 3, f"dy={d.get('dy')}")
    check("角度≈0", d.get("matched") and abs(d["angle_deg"]) <= 1, f"angle={d.get('angle_deg')}")
    # 校正图应与参考图基本相同(特征回到参考位置)
    if res.processed_image is not None:
        diff = cv2.absdiff(ref, res.processed_image)
        mean = diff.mean()
        check("校正图≈参考图", mean < 12, f"mean_diff={mean:.2f}")


def test_flip180():
    print("\n[2] 180°翻转校正")
    ref = make_ref_image()
    tool = build_tool(ref, (240, 190, 180, 190))
    flipped = cv2.rotate(ref, cv2.ROTATE_180)
    ctx = PipelineContext(original_image=flipped, current_image=flipped)
    res = tool.process(ctx)
    check("matched(翻转)", bool(res.passed), f"score={res.data.get('score')}")
    d = res.data
    check("角度≈180", d.get("matched") and abs(abs(d["angle_deg"]) - 180) <= 1,
          f"angle={d.get('angle_deg')}")
    if res.processed_image is not None:
        diff = cv2.absdiff(ref, res.processed_image)
        check("校正图≈参考图(翻转)", diff.mean() < 15, f"mean_diff={diff.mean():.2f}")


def test_miss():
    print("\n[3] 无特征(应NG)")
    ref = make_ref_image()
    tool = build_tool(ref, (240, 190, 180, 190))
    other = np.full((900, 1200, 3), (200, 200, 200), dtype=np.uint8)
    ctx = PipelineContext(original_image=other, current_image=other)
    res = tool.process(ctx)
    check("未匹配判NG", bool(res.passed) is False, f"score={res.data.get('score')}")


def test_pipeline_integration():
    print("\n[4] 流水线集成(位置修正 → MultiROI → 颜色识别)")
    ref = make_ref_image()
    # 参考方案:在板上画一个胶垫区(600,450 圆附近)
    pad_box = (520, 370, 160, 160)
    # 画胶:橙色块填满 pad_box 部分 → 覆盖率约 60%
    hsv = np.array([[[30, 180, 200]]], np.uint8)
    bgr = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
    cv2.rectangle(ref, (pad_box[0] + 30, pad_box[1] + 30),
                  (pad_box[0] + pad_box[2] - 30, pad_box[1] + pad_box[3] - 30),
                  bgr, -1)
    tool_pos = build_tool(ref, (240, 190, 180, 190))
    # 手工构造流水线: PositionCorrect -> MultiROI -> ColorRecognition
    from vision.pipeline import Pipeline, create_tool
    pipe = Pipeline(name="集成测试")
    pipe.add_step(tool_pos)
    multi = create_tool("MultiROI", {
        "regions": [{"name": "PAD", "x": pad_box[0], "y": pad_box[1],
                     "width": pad_box[2], "height": pad_box[3], "enabled": True}],
        "use_percentage": False,
    })
    pipe.add_step(multi)
    color = create_tool("ColorRecognition", {
        "color_name": "胶", "color_space": "HSV",
        "color_model": {"name": "胶", "color_space": "HSV",
                        "center": [30, 180, 200], "match_mode": "range",
                        "tolerance": [10, 60, 60],
                        "distance_threshold": 40.0,
                        "cluster_centers": [],
                        "normalize_illumination": False,
                        "adaptive_threshold": False, "source": "custom"},
        "match_mode": "range", "distance_threshold": 40.0,
        "normalize_illumination": False, "adaptive_threshold": False,
        "pass_min": 20.0, "pass_max": 90.0,
        "_input_source": "region:PAD",
    })
    pipe.add_step(color)

    # 输入:整体向右下平移 100,70 的图像(胶垫也应随板移动)
    M = np.float32([[1, 0, 100], [0, 1, 70]])
    moved = cv2.warpAffine(ref, M, (ref.shape[1], ref.shape[0]),
                           borderValue=(60, 70, 80))
    ok, results, _, _ = pipe.execute(moved)
    check("整板通过(板已平移仍检测到胶)", bool(ok),
          f"failed={[r.message for r in results if not bool(r.passed)]}")
    # 无位置修正时同输入应 NG(固定ROI裁空)
    pipe2 = Pipeline(name="无修正")
    pipe2.add_step(create_tool("MultiROI", {
        "regions": [{"name": "PAD", "x": pad_box[0], "y": pad_box[1],
                     "width": pad_box[2], "height": pad_box[3], "enabled": True}],
        "use_percentage": False}))
    pipe2.add_step(color)
    ok2, results2, _, _ = pipe2.execute(moved)
    check("对照组(无修正)NG", bool(ok2) is False,
          f"unexpected pass={[r.message for r in results2 if bool(r.passed)]}")


if __name__ == "__main__":
    test_shift()
    test_flip180()
    test_miss()
    test_pipeline_integration()
    print(f"\n结果: FAIL={FAIL}")
    sys.exit(1 if FAIL else 0)
