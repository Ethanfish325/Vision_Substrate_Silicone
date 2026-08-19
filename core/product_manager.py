# -*- coding: utf-8 -*-
"""
产品配置管理器
============
管理产品型号的配置文件，支持加载/保存/切换产品配置。

产品配置文件存放在 data/products/ 目录下，每个产品一个 JSON 文件。
工程师在设计模式中通过 UI 创建和编辑产品配置，
操作员在自动化模式中只需从下拉列表选择产品。

产品配置包含:
    - 相机参数 (曝光、增益、白平衡)
    - 运动参数 (轴号、速度、原点)
    - 检测位置列表 (每个位置有名称、坐标、关联的视觉方案)
    - DI触发参数
"""

import json
import os
from typing import List, Dict, Any, Optional

from core.paths import DATA_DIR
from core.log_manager import log_info, log_error, log_warning


PRODUCTS_DIR = os.path.join(DATA_DIR, 'products')


def ensure_products_dir():
    """确保产品配置目录存在"""
    os.makedirs(PRODUCTS_DIR, exist_ok=True)


def list_products() -> List[str]:
    """列出所有可用的产品型号名称

    Returns:
        List[str]: 产品名称列表（不含 .json 后缀）
    """
    ensure_products_dir()
    products = []
    try:
        for f in os.listdir(PRODUCTS_DIR):
            if f.endswith('.json'):
                products.append(f[:-5])  # 去掉 .json 后缀
    except OSError as e:
        log_error(f"读取产品配置目录失败: {e}")
    return sorted(products)


def load_product(name: str) -> Optional[Dict[str, Any]]:
    """加载指定产品型号的配置

    Args:
        name: 产品型号名称（不含 .json 后缀）

    Returns:
        Optional[Dict]: 产品配置字典，加载失败返回 None
    """
    filepath = os.path.join(PRODUCTS_DIR, f"{name}.json")
    if not os.path.exists(filepath):
        log_error(f"产品配置不存在: {name}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except (json.JSONDecodeError, OSError) as e:
        log_error(f"加载产品配置失败 [{name}]: {e}")
        return None


def save_product(config: Dict[str, Any]) -> bool:
    """保存产品配置

    Args:
        config: 产品配置字典，必须包含 "name" 字段

    Returns:
        bool: 是否保存成功
    """
    ensure_products_dir()
    name = config.get("name", "")
    if not name:
        log_error("保存产品配置失败: 缺少 name 字段")
        return False

    filepath = os.path.join(PRODUCTS_DIR, f"{name}.json")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        log_info(f"产品配置已保存: {name}")
        return True
    except OSError as e:
        log_error(f"保存产品配置失败 [{name}]: {e}")
        return False


def delete_product(name: str) -> bool:
    """删除产品配置

    Args:
        name: 产品型号名称

    Returns:
        bool: 是否删除成功
    """
    filepath = os.path.join(PRODUCTS_DIR, f"{name}.json")
    if not os.path.exists(filepath):
        log_warning(f"产品配置不存在，无法删除: {name}")
        return False
    try:
        os.remove(filepath)
        log_info(f"产品配置已删除: {name}")
        return True
    except OSError as e:
        log_error(f"删除产品配置失败 [{name}]: {e}")
        return False


def create_default_config(name: str) -> Dict[str, Any]:
    """创建默认产品配置模板

    Args:
        name: 产品型号名称

    Returns:
        Dict: 默认产品配置
    """
    return {
        "name": name,
        "description": "",

        "barcode_scan": {
            "enabled": False,
            "position": 0,
            "command": "01 54 04",
            "timeout_ms": 5000
        },

        "camera": {
            "exposure_time": 18000,
            "gain": 0,
            "white_balance": {"red": 1.0, "green": 1.0, "blue": 1.0}
        },

        "motion": {
            "axis": 1,
            "v_max": 50000,
            "a_max": 100000,
            "origin_position": 0,
            "move_timeout_s": 10
        },

        "positions": [
            {
                "name": "位置1",
                "position": 10000,
                "scheme": ""
            }
        ],

        "di_bit": 3,
        "poll_interval_ms": 50
    }
