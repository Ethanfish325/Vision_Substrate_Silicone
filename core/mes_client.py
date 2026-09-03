# -*- coding: utf-8 -*-
"""
MES 客户端模块
============
封装与 MES 服务器的 HTTP 接口通信，依据《didi mes接口文档(922024).pdf》实现：

1. 站位检测 CheckStation
    POST /openApi/didi/mes/CheckStation
    参数: sn(产品序列号), stationCode(站点编码)
    用途: 检测前校验该 SN 是否允许在当前站点过站（如是否已归属工单、是否重复过站）

2. 过站 SetStation
    POST /openApi/didi/mes/SetStation
    参数: sn, operator(操作员), stationCode, status(PASS/FAIL),
          failItem(FAIL时必填), prodno(工单号,可选), remark(备注,可选)
    用途: 检测完成后上报该 SN 的过站结果

地址: IP 172.16.100.18, 端口 7010
"""
import json
import time
from typing import Optional, Dict, Any

import requests

from core.log_manager import log_info, log_error, log_warning


class MESError(Exception):
    """MES 通信异常"""


class MESClient:
    """MES 服务器客户端。

    通过 HTTP POST 与 MES 服务器通信，提供站位检测与过站两个接口。
    所有请求均带超时，避免网络异常阻塞检测流程。
    """

    # 接口路径
    PATH_CHECK_STATION = "/openApi/didi/mes/CheckStation"
    PATH_SET_STATION = "/openApi/didi/mes/SetStation"

    def __init__(self, ip: str = "172.16.100.18", port: int = 7010,
                 station_code: str = "", operator: str = "",
                 timeout: float = 5.0):
        """初始化 MES 客户端。

        Args:
            ip: MES 服务器 IP
            port: MES 服务器端口
            station_code: 站点编码
            operator: 操作员/员工号
            timeout: 请求超时时间（秒）
        """
        self._ip = ip
        self._port = port
        self._station_code = station_code
        self._operator = operator
        self._timeout = timeout

    # ── 属性 ──

    @property
    def base_url(self) -> str:
        """拼接基础 URL。"""
        return f"http://{self._ip}:{self._port}"

    @property
    def station_code(self) -> str:
        return self._station_code

    @station_code.setter
    def station_code(self, value: str):
        self._station_code = value

    @property
    def operator(self) -> str:
        return self._operator

    @operator.setter
    def operator(self, value: str):
        self._operator = value

    # ── 通用请求 ──

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发送 POST JSON 请求并解析响应。

        Args:
            path: 接口路径
            payload: 请求体字典

        Returns:
            解析后的响应字典（含 code/data/msg）

        Raises:
            MESError: 网络异常、超时或响应格式错误
        """
        url = self.base_url + path
        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=self._timeout,
                headers={"Content-Type": "application/json"},
            )
        except requests.exceptions.Timeout:
            raise MESError(f"MES 请求超时: {url}")
        except requests.exceptions.ConnectionError as e:
            raise MESError(f"MES 连接失败: {url} ({e})")
        except requests.exceptions.RequestException as e:
            raise MESError(f"MES 请求异常: {url} ({e})")

        try:
            data = resp.json()
        except (ValueError, json.JSONDecodeError):
            raise MESError(f"MES 响应非 JSON: {resp.status_code} {resp.text[:200]}")

        if not isinstance(data, dict):
            raise MESError(f"MES 响应格式错误: {data}")

        return data

    # ── 站位检测 ──

    def check_station(self, sn: str) -> Dict[str, Any]:
        """站位检测：校验 SN 是否允许在当前站点过站。

        Args:
            sn: 产品序列号

        Returns:
            响应字典（含 code/data/msg）

        Raises:
            MESError: 通信异常
        """
        payload = {
            "sn": sn,
            "stationCode": self._station_code,
        }
        log_info(f"MES 站位检测 CheckStation: sn={sn}, stationCode={self._station_code}")
        data = self._post(self.PATH_CHECK_STATION, payload)
        code = data.get("code")
        if code is not None and int(code) != 200:
            msg = data.get("msg", "") or data.get("data", {}).get("oErrMessage", "")
            log_warning(f"MES 站位检测未通过: sn={sn}, code={code}, msg={msg}")
        else:
            log_info(f"MES 站位检测通过: sn={sn}, code={code}")
        return data

    # ── 过站 ──

    def set_station(self, sn: str, status: str,
                    fail_item: str = "", prodno: str = "",
                    remark: str = "") -> Dict[str, Any]:
        """过站：上报 SN 的检测结果。

        Args:
            sn: 产品序列号
            status: 状态，PASS 或 FAIL
            fail_item: 失败项（FAIL 时必填，按需求 PASS 填 "OK"，FAIL 填 "NG"）
            prodno: 工单号（可选）
            remark: 备注（可选）

        Returns:
            响应字典（含 code/data/msg）

        Raises:
            MESError: 通信异常
        """
        status = status.upper()
        if status not in ("PASS", "FAIL"):
            raise MESError(f"过站状态非法: {status}（应为 PASS/FAIL）")

        payload = {
            "sn": sn,
            "operator": self._operator,
            "stationCode": self._station_code,
            "status": status,
        }
        # failItem 按需求始终填写：PASS 填 "OK"，FAIL 填 "NG"
        payload["failItem"] = fail_item or ("OK" if status == "PASS" else "NG")
        if prodno:
            payload["prodno"] = prodno
        if remark:
            payload["remark"] = remark

        log_info(f"MES 过站 SetStation: sn={sn}, status={status}, "
                 f"operator={self._operator}, stationCode={self._station_code}")
        data = self._post(self.PATH_SET_STATION, payload)
        code = data.get("code")
        if code is not None and int(code) != 200:
            msg = data.get("msg", "") or data.get("data", {}).get("oErrMessage", "")
            log_warning(f"MES 过站失败: sn={sn}, code={code}, msg={msg}")
        else:
            log_info(f"MES 过站成功: sn={sn}, code={code}")
        return data

    # ── 便捷方法 ──

    def report_pass(self, sn: str, prodno: str = "", remark: str = "") -> Dict[str, Any]:
        """上报 PASS（OK）结果。"""
        return self.set_station(sn, "PASS", fail_item="OK", prodno=prodno, remark=remark)

    def report_fail(self, sn: str, prodno: str = "", remark: str = "") -> Dict[str, Any]:
        """上报 FAIL（NG）结果。"""
        return self.set_station(sn, "FAIL", fail_item="NG", prodno=prodno, remark=remark)

    def test_connection(self) -> bool:
        """测试与 MES 服务器的连通性。

        通过一次轻量的站位检测（空 SN 或占位 SN）探测服务器是否可达。
        仅用于设置窗口的「测试连接」按钮，不依赖业务数据。

        Returns:
            bool: 服务器是否可达
        """
        try:
            # 使用占位 SN 探测，服务器可达即可（即使返回业务错误也算连通）
            self._post(self.PATH_CHECK_STATION, {
                "sn": "CONN_TEST",
                "stationCode": self._station_code or "TEST",
            })
            return True
        except MESError as e:
            log_warning(f"MES 连接测试失败: {e}")
            return False
