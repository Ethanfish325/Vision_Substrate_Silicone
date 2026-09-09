# -*- coding: utf-8 -*-
"""位置修正算子行为自测(合成图像,ROI 随动语义)。

架构说明(重构后):
    位置修正【不再把整图旋转/平移校正回参考姿态】,而是只计算并写入
    context.location(参考→当前的仿射矩阵/角度/锚点),由 MultiROI 把参考
    ROI 变换为当前图上的实际区域(ROI 随动),裁剪时局部摆正。
    因此 process 输出的 processed_image = 原始输入图(整图不被旋转),
    校验重点是:
      1) location 几何正确(锚点映射到特征实际位置、角度/平移量语义正确);
      2) 按 location 在原图上还原出的模板区域内容 == 模板(证明 ROI 随动
         能把 ROI 摆到真实产品特征上);
      3) 级联 MultiROI + 颜色识别后,胶垫覆盖率不受板卡偏移影响。
"""
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
    """画一块'板子':底色 + 一个独特角标特征(当作定位基准)。"""
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


def template_region_content(img_bgr, loc, templ_bgr, tol=10.0):
    """按 location 在原图上还原模板区域(ROI 随动语义),返回其与模板的灰度差异。

    位置修正声称"模板区域在当前图上位于 锚点±宽高、旋转 angle_deg",
    摆正裁剪后应当还原出模板本身——这是 ROI 随动正确性的直接证据。
    """
    from vision.geometry_util import crop_rotated_rect
    w, h = templ_bgr.shape[1], templ_bgr.shape[0]
    crop = crop_rotated_rect(img_bgr,
                             float(loc["cur_anchor_x"]),
                             float(loc["cur_anchor_y"]),
                             w, h, float(loc["angle_deg"]))
    if crop.shape[0] != h or crop.shape[1] != w:
        crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    to_gray = (lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
               if im.ndim == 3 else im)
    a = to_gray(crop).astype(np.float32)
    b = to_gray(templ_bgr).astype(np.float32)
    return float(np.abs(a - b).mean())


def run_and_locate(tool, img):
    ctx = PipelineContext(original_image=img, current_image=img)
    res = tool.process(ctx)
    return res, (ctx.location if res.passed else None)


def test_shift():
    print("\n[1] 仅平移校正")
    ref = make_ref_image()
    tool = build_tool(ref, (240, 190, 180, 190))  # L角标整体
    # 特征参考中心
    ref_cx = 240 + 180 / 2.0   # 330
    ref_cy = 190 + 190 / 2.0   # 285
    templ = tool._get_template()
    # 向右下平移 80,60
    M = np.float32([[1, 0, 80], [0, 1, 60]])
    moved = cv2.warpAffine(ref, M, (ref.shape[1], ref.shape[0]),
                           borderValue=(60, 70, 80))
    res, loc = run_and_locate(tool, moved)
    check("matched", bool(res.passed), f"score={res.data.get('score')}")
    d = res.data
    check("dx≈+80(特征右移)", d.get("matched") and abs(d["dx"] - 80) < 3, f"dx={d.get('dx')}")
    check("dy≈+60(特征下移)", d.get("matched") and abs(d["dy"] - 60) < 3, f"dy={d.get('dy')}")
    check("角度≈0", d.get("matched") and abs(d["angle_deg"]) <= 1, f"angle={d.get('angle_deg')}")
    # ROI 随动语义:整图不再被旋转/平移校正,processed_image 就是原始输入
    if res.processed_image is not None:
        raw_diff = cv2.absdiff(res.processed_image, moved).mean()
        check("processed_image = 原始输入(整图不旋转)", raw_diff < 1,
              f"mean_diff={raw_diff:.2f}")
    # location 锚点应落在"平移后特征的真实位置"
    if loc:
        check("锚点≈平移后特征中心(410,345)",
              abs(loc["cur_anchor_x"] - (ref_cx + 80)) < 3
              and abs(loc["cur_anchor_y"] - (ref_cy + 60)) < 3,
              f"anchor=({loc['cur_anchor_x']:.1f},{loc['cur_anchor_y']:.1f})")
        diff = template_region_content(moved, loc, templ)
        check("按 location 还原模板区≈模板(ROI 随动正确)", diff < 10,
              f"mean_diff={diff:.2f}")


def test_flip180():
    print("\n[2] 180°翻转校正")
    ref = make_ref_image()
    tool = build_tool(ref, (240, 190, 180, 190))
    ref_cx = 240 + 180 / 2.0
    ref_cy = 190 + 190 / 2.0
    templ = tool._get_template()
    flipped = cv2.rotate(ref, cv2.ROTATE_180)
    # 180° 翻转(中心对称):(x,y) -> (w-1-x, h-1-y),特征中心随之
    exp_x = flipped.shape[1] - 1 - ref_cx   # 869
    exp_y = flipped.shape[0] - 1 - ref_cy   # 614
    res, loc = run_and_locate(tool, flipped)
    check("matched(翻转)", bool(res.passed), f"score={res.data.get('score')}")
    d = res.data
    check("角度≈180", d.get("matched") and abs(abs(d["angle_deg"]) - 180) <= 1,
          f"angle={d.get('angle_deg')}")
    if res.processed_image is not None:
        raw_diff = cv2.absdiff(res.processed_image, flipped).mean()
        check("processed_image = 原始输入(整图不旋转)", raw_diff < 1,
              f"mean_diff={raw_diff:.2f}")
    if loc:
        check("锚点≈翻转后特征中心",
              abs(loc["cur_anchor_x"] - exp_x) < 3
              and abs(loc["cur_anchor_y"] - exp_y) < 3,
              f"anchor=({loc['cur_anchor_x']:.1f},{loc['cur_anchor_y']:.1f})")
        diff = template_region_content(flipped, loc, templ)
        check("按 location 还原模板区≈模板(ROI 随动正确)", diff < 10,
              f"mean_diff={diff:.2f}")


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
    # 位置修正步骤应报告随动角度≈0(仅平移)
    pos_data = results[0].data if results else {}
    check("位置修正 matched 且角度≈0", bool(pos_data.get("matched"))
          and abs(pos_data.get("angle_deg", 999)) <= 1,
          f"data={pos_data}")
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
