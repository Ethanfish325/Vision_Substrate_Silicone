# 通用条码识别算子使用说明

## 概述

`QRCodeRecognize` 算子已升级为**通用条码识别算子**，在保持原有二维码（QR Code）识别能力完全不变的前提下，新增一维码（1D Barcode）识别功能。

- **二维码识别**：使用 OpenCV `QRCodeDetector`（多策略识别，保持原有逻辑）
- **一维码识别**：使用 `pyzbar`（支持 Code 128、Code 39、EAN-13/UPC-A 等常用格式）
- **自动区分**：二维码与一维码自动区分，输出条码类型（QR/1D）
- **混合识别**：同一图像中同时存在二维码和一维码时，全部识别并分别输出
- **无条码**：返回空结果，不抛异常

## 依赖

```bash
pip install pyzbar
```

> 注意：pyzbar 依赖系统 ZBar 库。Windows 下 `pip install pyzbar` 会自动安装预编译的 ZBar DLL。

## 参数配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `require_pass` | bool | True | 是否将"识别到条码"作为通过条件 |
| `expected_prefix` | str | "" | 可选，期望的 SN 前缀（用于校验） |
| `enable_1d` | bool | True | 是否启用一维码识别 |
| `barcode_formats` | list | `["CODE_128", "CODE_39", "EAN_13", "UPC_A"]` | 可选的一维码格式集合（空表示全部格式） |

### 一维码格式

| 配置名 | pyzbar 类型 | 说明 |
|--------|------------|------|
| `CODE_128` | CODE128 | Code 128 |
| `CODE_39` | CODE39 | Code 39 |
| `EAN_13` | EAN13 | EAN-13 |
| `EAN_8` | EAN8 | EAN-8 |
| `UPC_A` | UPCA | UPC-A |
| `UPC_E` | UPCE | UPC-E |
| `ITF` | I25 | 交错 2/5 |
| `CODABAR` | CODABAR | Codabar |

## 输出数据

`ToolResult.data` 包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `qr_data` | str | 第一个识别到的条码内容（向后兼容，供 SN 保存） |
| `recognized` | bool | 是否识别到条码 |
| `barcodes` | list | 所有识别到的条码详情列表 |
| `barcode_count` | int | 识别到的条码数量 |

`barcodes` 中每个元素：

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | str | 条码类型（`QR` 或 `1D`） |
| `data` | str | 条码内容 |
| `confidence` | float | 识别置信度（0-1） |
| `barcode_type` | str | 一维码具体格式（如 `CODE128`），二维码无此字段 |
| `bbox` | tuple | 条码位置矩形框 `(x, y, w, h)` |

## 使用示例

```python
from vision.tools.recognize import QRCodeRecognize
from vision.tools.base_tool import PipelineContext

tool = QRCodeRecognize({
    "require_pass": True,
    "expected_prefix": "",
    "enable_1d": True,
    "barcode_formats": ["CODE_128", "CODE_39", "EAN_13", "UPC_A"],
})

ctx = PipelineContext(original_image=img, current_image=img)
result = tool.process(ctx)

print(result.data["qr_data"])       # 第一个条码内容
print(result.data["barcodes"])      # 所有条码详情
for bc in result.data["barcodes"]:
    print(bc["type"], bc["data"], bc["confidence"], bc["bbox"])
```

## 向后兼容

- 类名保持 `QRCodeRecognize`，现有方案配置无需修改
- `data["qr_data"]` 字段保持原有语义（第一个条码内容），供 SN 保存逻辑使用
- `require_pass` / `expected_prefix` 参数保持原有行为
- 新增参数（`enable_1d`、`barcode_formats`）有默认值，不配置也能正常工作

## 测试

运行单元测试：

```bash
python tests/test_barcode_recognize.py
```

覆盖场景：
1. 纯二维码
2. 纯一维码（Code 128 / Code 39）
3. 二维码 + 一维码混合
4. 模糊/倾斜/低质量条码（不抛异常）
5. 无条码（返回空结果）
6. 禁用一维码
7. 一维码格式过滤
8. SN 前缀校验
