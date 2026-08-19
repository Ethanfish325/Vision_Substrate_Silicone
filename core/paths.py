# -*- coding: utf-8 -*-

import os
import sys


def _get_data_dir() -> str:
    if getattr(sys, 'frozen', False):
        # 打包后：先尝试 exe 同级目录（用于便携式部署），
        # 如果不存在则使用 _internal/data（PyInstaller 打包的数据目录）
        base_dir = os.path.dirname(sys.executable)
        data_dir = os.path.join(base_dir, 'data')
        if os.path.isdir(data_dir):
            return data_dir
        # 回退到 _internal/data
        internal_dir = os.path.join(base_dir, '_internal')
        data_dir = os.path.join(internal_dir, 'data')
        if os.path.isdir(data_dir):
            return data_dir
        # 最终回退
        return os.path.join(base_dir, 'data')
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'data')


DATA_DIR = _get_data_dir()
SCHEME_DIR = os.path.join(DATA_DIR, 'schemes')
USERS_FILE = os.path.join(DATA_DIR, 'users.json')
CONFIG_FILE = os.path.join(DATA_DIR, 'config.json')
ERRORS_DIR = os.path.join(DATA_DIR, 'errors')
LOGS_DIR = os.path.join(DATA_DIR, 'logs')
ICON_FILE = os.path.join(DATA_DIR, 'icon.png')

# 生产数据目录（按日期/OK-NG 组织）
PRODUCTION_DATA_DIR = os.path.join(DATA_DIR, 'production data')


def ensure_dirs():
    for d in [DATA_DIR, SCHEME_DIR, LOGS_DIR, PRODUCTION_DATA_DIR]:
        os.makedirs(d, exist_ok=True)
