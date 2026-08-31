# -*- coding: utf-8 -*-
"""
通用条码识别算子单元测试
========================
覆盖场景：
    1. 纯二维码（QR Code）
    2. 纯一维码（Code 128 / Code 39 / EAN-13）
    3. 二维码 + 一维码混合
    4. 模糊/倾斜/低质量条码
    5. 无条码（返回空结果，不抛异常）

运行:
    python -m pytest tests/test_barcode_recognize.py -v
    或
    python tests/test_barcode_recognize.py
"""
import io
import sys
import os
import cv2
import numpy as np

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.tools.recognize import QRCodeRecognize
from vision.tools.base_tool import PipelineContext


def make_qr_image(data, size=300):
    """生成二维码图像。"""
    qr = cv2.QRCodeEncoder_create()
    img = qr.encode(data)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_1d_image(data, barcode_type='code128'):
    """用 python-barcode 生成一维码图像。"""
    import barcode
    from barcode.writer import ImageWriter
    writer = ImageWriter()
    writer.set_options({'module_width': 0.4, 'module_height': 15,
                        'font_size': 10, 'text_distance': 2,
                        'quiet_zone': 6.0})
    bc = barcode.get(barcode_type, data, writer=writer)
    buf = io.BytesIO()
    bc.write(buf)
    buf.seek(0)
    return cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)


def run_tool(img, params=None):
    """运行条码识别算子。"""
    tool = QRCodeRecognize(params)
    ctx = PipelineContext(original_image=img, current_image=img)
    return tool.process(ctx)


def test_pure_qr():
    """纯二维码识别。"""
    img = make_qr_image('QR-SN-001-ABCDEF')
    result = run_tool(img)
    qr_data = result.data.get('qr_data')
    assert qr_data == 'QR-SN-001-ABCDEF', f"qr_data={qr_data!r}"
    assert result.data.get('recognized') is True
    barcodes = result.data.get('barcodes', [])
    assert len(barcodes) == 1, f"barcodes={barcodes}"
    assert barcodes[0]['type'] == 'QR'
    assert barcodes[0]['data'] == 'QR-SN-001-ABCDEF'
    assert barcodes[0]['confidence'] > 0
    assert len(barcodes[0]['bbox']) == 4
    print("[PASS] 纯二维码识别")


def test_pure_1d_code128():
    """纯一维码（Code 128）识别。"""
    img = make_1d_image('1234567890', 'code128')
    result = run_tool(img)
    assert result.data.get('qr_data') == '1234567890'
    assert result.data.get('recognized') is True
    barcodes = result.data.get('barcodes', [])
    assert len(barcodes) == 1
    assert barcodes[0]['type'] == '1D'
    assert barcodes[0]['data'] == '1234567890'
    assert barcodes[0]['barcode_type'] == 'CODE128'
    assert len(barcodes[0]['bbox']) == 4
    print("[PASS] 纯一维码 Code128 识别")


def test_pure_1d_code39():
    """纯一维码（Code 39）识别。"""
    img = make_1d_image('ABC-123', 'code39')
    result = run_tool(img)
    assert result.data.get('recognized') is True
    barcodes = result.data.get('barcodes', [])
    assert len(barcodes) >= 1
    assert barcodes[0]['type'] == '1D'
    print("[PASS] 纯一维码 Code39 识别")


def test_mixed_qr_and_1d():
    """二维码 + 一维码混合识别。"""
    qr_img = make_qr_image('QR-MIX-001')
    code_img = make_1d_image('9876543210', 'code128')
    # 拼接两张图（左右并排）
    h = max(qr_img.shape[0], code_img.shape[0])
    canvas = np.full((h, qr_img.shape[1] + code_img.shape[1], 3), 255, dtype=np.uint8)
    canvas[:qr_img.shape[0], :qr_img.shape[1]] = qr_img
    canvas[:code_img.shape[0], qr_img.shape[1]:] = code_img
    result = run_tool(canvas)
    barcodes = result.data.get('barcodes', [])
    types = {bc['type'] for bc in barcodes}
    assert 'QR' in types
    assert '1D' in types
    print("[PASS] 二维码 + 一维码混合识别")


def test_blurred_barcode():
    """模糊条码识别。"""
    img = make_1d_image('555666777', 'code128')
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    result = run_tool(blurred)
    # 模糊后可能识别失败，但不应抛异常
    assert result.success is True
    print("[PASS] 模糊条码（不抛异常）")


def test_rotated_barcode():
    """倾斜条码识别。"""
    img = make_1d_image('111222333', 'code128')
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    result = run_tool(rotated)
    # 倾斜后可能识别失败，但不应抛异常
    assert result.success is True
    print("[PASS] 倾斜条码（不抛异常）")


def test_no_barcode():
    """无条码图像。"""
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    result = run_tool(img)
    assert result.data.get('qr_data') == ''
    assert result.data.get('recognized') is False
    assert result.data.get('barcodes') == []
    assert result.success is True
    print("[PASS] 无条码（返回空结果，不抛异常）")


def test_disable_1d():
    """禁用一维码识别。"""
    img = make_1d_image('1234567890', 'code128')
    result = run_tool(img, {'enable_1d': False})
    assert result.data.get('recognized') is False
    assert result.data.get('barcodes') == []
    print("[PASS] 禁用一维码识别")


def test_format_filter():
    """一维码格式过滤（只允许 EAN_13，Code128 应被过滤）。"""
    img = make_1d_image('1234567890', 'code128')
    result = run_tool(img, {'barcode_formats': ['EAN_13']})
    assert result.data.get('recognized') is False
    print("[PASS] 一维码格式过滤")


def test_expected_prefix():
    """SN 前缀校验。"""
    img = make_qr_image('SN-ABC-123')
    result = run_tool(img, {'expected_prefix': 'SN-'})
    assert result.data.get('recognized') is True
    # 前缀不匹配
    result2 = run_tool(img, {'expected_prefix': 'XX-'})
    assert result2.data.get('recognized') is False
    print("[PASS] SN 前缀校验")


def run_all():
    """运行所有测试。"""
    tests = [
        test_pure_qr,
        test_pure_1d_code128,
        test_pure_1d_code39,
        test_mixed_qr_and_1d,
        test_blurred_barcode,
        test_rotated_barcode,
        test_no_barcode,
        test_disable_1d,
        test_format_filter,
        test_expected_prefix,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {t.__name__}: {e}")
    print(f"\n通过 {passed}/{len(tests)} 项测试")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
