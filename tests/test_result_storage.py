# -*- coding: utf-8 -*-
"""测试 ResultStorage 新的保存逻辑（需求1）。

验证:
    1. 一个 SN 一个文件夹，照片与 XML 一同保存
    2. OK 与 NG 都保存
    3. 未识别到 SN 号时保存到「未识别到SN号」文件夹
    4. 每个 SN 一个 XML，格式符合要求
    5. update_position_result 能更新 XML 判定
"""
import os
import shutil
import sys
import tempfile

# 设置 stdout 编码为 utf-8，避免中文输出在 GBK 终端报错
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np

from core.result_storage import ResultStorage, UNKNOWN_SN_DIR


def _make_image():
    """生成一张测试图像"""
    return np.zeros((100, 200, 3), dtype=np.uint8)


def test_save_position_data():
    """测试保存单个点位数据"""
    # 使用临时目录作为生产数据目录
    tmp = tempfile.mkdtemp()
    try:
        storage = ResultStorage()
        storage._production_dir = tmp

        # 保存一个 OK 点位（有 SN）
        info = storage.save_position_data(
            scheme_name="DX8000_PCBA",
            sn="SA16179570",
            position_name="1.1",
            position_index=1,
            annotated_image=_make_image(),
            passed=True,
        )
        assert info is not None, "保存失败"
        assert os.path.exists(info['img_path']), "照片未保存"
        assert os.path.exists(info['xml_path']), "XML 未保存"

        # 验证目录结构: tmp/YYYY-MM-DD/SA16179570/
        date_dir = os.path.dirname(info['sn_dir'])
        assert os.path.basename(info['sn_dir']) == "SA16179570", \
            f"SN 文件夹名错误: {os.path.basename(info['sn_dir'])}"

        # 验证照片文件名含 OK
        assert "_OK.jpg" in os.path.basename(info['img_path']), \
            f"照片文件名应含 OK: {os.path.basename(info['img_path'])}"

        # 验证 XML 内容
        with open(info['xml_path'], 'r', encoding='utf-8') as f:
            xml = f.read()
        assert '<?xml version="1.0" encoding="UTF-8"?>' in xml, "XML 头错误"
        assert '<tests>' in xml and '</tests>' in xml, "XML tests 标签错误"
        assert 'test_sn="SA16179570"' in xml, "XML test_sn 错误"
        assert 'test_result="OK"' in xml, "XML test_result 错误"
        assert 'imgurl=' in xml, "XML imgurl 缺失"
        print("[PASS] 保存 OK 点位（有SN）成功，XML 格式正确")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_save_ng_and_unknown_sn():
    """测试保存 NG 点位 + 未识别到 SN 号"""
    tmp = tempfile.mkdtemp()
    try:
        storage = ResultStorage()
        storage._production_dir = tmp

        # 保存一个 NG 点位（有 SN）
        info_ng = storage.save_position_data(
            scheme_name="DX8000_PCBA",
            sn="SA16179571",
            position_name="2.1",
            position_index=5,
            annotated_image=_make_image(),
            passed=False,
        )
        assert "_NG.jpg" in os.path.basename(info_ng['img_path']), \
            f"NG 照片文件名应含 NG: {os.path.basename(info_ng['img_path'])}"
        with open(info_ng['xml_path'], 'r', encoding='utf-8') as f:
            assert 'test_result="NG"' in f.read(), "NG XML test_result 错误"
        print("[PASS] 保存 NG 点位（有SN）成功")

        # 保存一个未识别到 SN 的点位
        info_unknown = storage.save_position_data(
            scheme_name="DX8000_PCBA",
            sn="",
            position_name="3.1",
            position_index=9,
            annotated_image=_make_image(),
            passed=False,
        )
        assert os.path.basename(info_unknown['sn_dir']) == UNKNOWN_SN_DIR, \
            f"未识别SN应保存到 {UNKNOWN_SN_DIR}: {os.path.basename(info_unknown['sn_dir'])}"
        assert os.path.exists(info_unknown['img_path']), "未识别SN照片未保存"
        assert os.path.exists(info_unknown['xml_path']), "未识别SN XML 未保存"
        print(f"[PASS] 保存未识别SN点位成功，目录: {UNKNOWN_SN_DIR}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_update_position_result():
    """测试更新 XML 判定"""
    tmp = tempfile.mkdtemp()
    try:
        storage = ResultStorage()
        storage._production_dir = tmp

        # 先保存一个 NG 点位
        info = storage.save_position_data(
            scheme_name="DX8000_PCBA",
            sn="SA16179572",
            position_name="1.2",
            position_index=2,
            annotated_image=_make_image(),
            passed=False,
        )

        # 更新为 OK
        updated = storage.update_position_result(sn="SA16179572", passed=True)
        assert updated is not None, "更新 XML 失败"
        with open(updated, 'r', encoding='utf-8') as f:
            xml = f.read()
        assert 'test_result="OK"' in xml, "更新后 test_result 应为 OK"
        assert 'test_sn="SA16179572"' in xml, "更新后 test_sn 应保留"
        print("[PASS] 更新 XML 判定成功")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_save_position_data()
    test_save_ng_and_unknown_sn()
    test_update_position_result()
    print("\n[PASS] 全部测试通过")
