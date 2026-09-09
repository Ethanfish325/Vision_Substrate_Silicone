# -*- coding: utf-8 -*-
"""ROI 随动集成自测:位置修正 → MultiROI(随动) → 颜色识别。

场景:产品(参考图上含角标特征 + 胶垫条)整体平移 / 旋转 180 / 旋转 90 /
小角度旋转(任意角度模式)。位置修正算出 M_ref2cur,MultiROI 将参考 ROI
变换到当前图,base_tool 摆正裁剪后交给颜色识别。期望胶垫覆盖率稳定
(≈ 参考值),即 ROI 真正"随动"到了产品旋转/平移后的位置。
"""
import sys
sys.path.insert(0, r"D:\VScode Project\Python project\Vision_Substrate_Silicone")

import numpy as np
import cv2

from vision.pipeline import Pipeline, create_tool
from vision.tools.position import PositionCorrect

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


def make_ref(w=1400, h=1000):
    """板+左上角标(L形)+长条胶垫(颜色唯一,横置,方向可辨)。"""
    img = np.full((h, w, 3), (55, 65, 75), np.uint8)
    cv2.rectangle(img, (200, 150), (1200, 850), (150, 160, 170), -1)
    # 角标(L形,独特)
    cv2.rectangle(img, (240, 180), (400, 215), (20, 25, 35), -1)
    cv2.rectangle(img, (240, 180), (275, 360), (20, 25, 35), -1)
    # 右下小方块(破对称,帮助角度判定)
    cv2.rectangle(img, (1130, 780), (1185, 835), (20, 25, 35), -1)
    return img


def glue_ratio(img, box):
    x, y, w, h = box
    roi = img[y:y + h, x:x + w]
    m = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV),
                    np.array([18, 120, 120]), np.array([42, 255, 255]))
    return float(m.mean() / 255.0) * 100.0


def place_glue(ref, cx, cy, w=160, h=30):
    """在参考图 (cx,cy) 处放一条横置胶。"""
    hsv = np.array([[[30, 180, 200]]], np.uint8)
    bgr = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    cv2.rectangle(ref, (x0, y0), (x0 + w, y0 + h), bgr, -1)
    return (x0, y0, w, h)


def make_variant(ref, ang_deg, dx=0.0, dy=0.0):
    """生成当前图:参考图绕中心旋转 ang_deg 再平移 (dx,dy)。"""
    h, w = ref.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang_deg, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    out = cv2.warpAffine(ref, M, (w, h), borderValue=(55, 65, 75))
    return out, M


def build_pipe(ref, pad_box, rot_mode="0and180"):
    """位置修正 + MultiROI(参考姿态 pad_box)+ 颜色识别。"""
    tool_pos = PositionCorrect()
    ok = tool_pos.set_template(ref[180:360, 240:400].copy(), 240, 180)
    assert ok, "模板应有效"
    tool_pos.params["rot_mode"] = rot_mode
    tool_pos.params["threshold"] = 0.6
    pw, ph = pad_box[2], pad_box[3]
    pipe = Pipeline("随动")
    pipe.add_step(tool_pos)
    pipe.add_step(create_tool("MultiROI", {
        "regions": [{"name": "PAD", "x": pad_box[0], "y": pad_box[1],
                     "width": pw, "height": ph, "enabled": True}],
        "use_percentage": False}))
    pipe.add_step(create_tool("ColorRecognition", {
        "color_model": {"name": "胶", "color_space": "HSV",
                        "center": [30, 180, 200], "match_mode": "range",
                        "tolerance": [10, 60, 60], "distance_threshold": 40.0,
                        "cluster_centers": [], "normalize_illumination": False,
                        "adaptive_threshold": False, "source": "custom"},
        "match_mode": "range", "distance_threshold": 40.0,
        "normalize_illumination": False, "adaptive_threshold": False,
        "pass_min": 15.0, "pass_max": 95.0, "_input_source": "region:PAD",
    }))
    return pipe


def run_case(name, ref, pad_center, ang_deg, dx, dy, rot_mode, pw=200, ph=40):
    """验证:变体图(平移/翻转/旋转)经 ROI 随动后胶垫覆盖率 ≈ 参考值。"""
    img, Mv = make_variant(ref, ang_deg, dx, dy)
    pad_box = (int(pad_center[0] - pw / 2), int(pad_center[1] - ph / 2), pw, ph)
    pipe = build_pipe(ref, pad_box, rot_mode)
    # 参考覆盖率:直接在参考图上测同一 ROI(胶垫画在中心,占 ~60%)
    ref_cov = glue_ratio(ref, pad_box)

    ok, results, _, _ = pipe.execute(img)
    cov = None
    followed = None
    for r in results:
        d = getattr(r, "data", {}) or {}
        if d.get("area_ratio") is not None:
            cov = d["area_ratio"]
        if d.get("followed") is not None:
            followed = d["followed"]
    check(f"[{name}] 检测通过", bool(ok),
          f"failed={[r.message for r in results if not bool(r.passed)]}")
    check(f"[{name}] MultiROI 已随动", bool(followed), f"followed={followed}")
    if cov is not None:
        check(f"[{name}] 覆盖率接近参考({ref_cov:.0f}%)", abs(cov - ref_cov) < 25,
              f"cov={cov:.1f}% ref={ref_cov:.1f}%")
    else:
        check(f"[{name}] 得到覆盖率", False, "颜色识别未输出 area_ratio")


def main():
    w, h = 1400, 1000
    ref = make_ref(w, h)
    pad_center = (600, 600)
    pw, ph = 200, 40
    # 在参考图放胶:只画一条居中横置长条(160x30),ROI(200x40)覆盖率≈60%
    place_glue(ref, pad_center[0], pad_center[1], w=160, h=30)
    pad_box = (int(pad_center[0] - pw / 2), int(pad_center[1] - ph / 2), pw, ph)
    print("参考胶垫覆盖率:", round(glue_ratio(ref, pad_box), 1), "%")

    print("\n[1] 纯平移 (80,50)")
    run_case("平移", ref, pad_center, 0, 80, 50, "0and180")

    print("\n[2] 180° 翻转")
    run_case("翻转180", ref, pad_center, 180, 0, 0, "0and180")

    print("\n[3] 90° 旋转")
    run_case("旋转90", ref, pad_center, 90, 0, 0, "any")

    print("\n[4] 小角度 +7°")
    run_case("小角度7", ref, pad_center, 7, 0, 0, "any")

    print("\n[5] 小角度 -5°")
    run_case("小角度-5", ref, pad_center, -5, 0, 0, "any")

    print(f"\n结果: PASS={PASS} FAIL={FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
