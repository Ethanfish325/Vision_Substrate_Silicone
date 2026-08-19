# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict, Optional

from core.paths import CONFIG_FILE


class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._config = {}
        self._load()

    DEFAULT_CONFIG = {
        'camera': {
            'exposure_time': 30000,
            'gain': 1.0,
            'resolution_width': 1920,
            'resolution_height': 1080,
            'pixel_format': 'Mono8',
        },
        'system': {
            'language': 'zh-CN',
            'auto_login': False,
            'retention_days': 90,
            'log_max_size_gb': 50,
            'log_cleanup_ratio': 0.5,
        },
        'display': {
            'fullscreen': False,
            'worker_mode': True,
        }
    }

    def _load(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._config = {}
        else:
            self._config = {}
        self._config = self._merge_defaults(self.DEFAULT_CONFIG, self._config)

    def _merge_defaults(self, default: Dict, target: Dict) -> Dict:
        result = default.copy()
        for key, value in target.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_defaults(result[key], value)
            else:
                result[key] = value
        return result

    def save(self):
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, ensure_ascii=False, indent=2)

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any):
        keys = key_path.split('.')
        target = self._config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    def reset_to_defaults(self):
        self._config = self.DEFAULT_CONFIG.copy()
        self.save()
