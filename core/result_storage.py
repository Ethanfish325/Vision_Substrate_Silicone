# -*- coding: utf-8 -*-

import csv
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import cv2
import numpy as np

from core.paths import PRODUCTION_DATA_DIR


# 未识别到 SN 号时的统一文件夹名
UNKNOWN_SN_DIR = "未识别到SN号"


class ResultStorage:
    """生产数据存储管理器

    目录结构:
        data/production data/
            YYYY-MM-DD/
                {SN号}/
                    {SN号}_{HHMMSS}_pos{序号}_{位置名}_{OK/NG}.jpg  # 该点位照片
                    {SN号}.xml                                        # 该点位 MES 上传用 XML
                未识别到SN号/
                    {时间戳}_pos{序号}_{位置名}_{OK/NG}.jpg          # 未识别到 SN 的点位照片
                    {时间戳}.xml                                      # 未识别到 SN 的 XML

    说明:
        - 每个点位即一块独立板卡，拥有独立的 SN 号（QR 识别结果）。
        - 一个 SN 号对应一个文件夹，该点位照片与 XML 一同保存。
        - 测试结果 OK 与 NG 都保存（照片文件名含判定）。
        - 未识别到 SN 号时，统一保存到「未识别到SN号」文件夹。
    """

    def __init__(self):
        self._production_dir = PRODUCTION_DATA_DIR
        os.makedirs(self._production_dir, exist_ok=True)

    # ── 公共保存接口 ──

    def save_position_data(self, scheme_name: str, sn: str,
                           position_name: str, position_index: int,
                           annotated_image: np.ndarray, passed: bool,
                           timestamp: Optional[str] = None) -> Optional[dict]:
        """保存单个点位（板卡）的数据：照片 + XML。

        每个点位即一块独立板卡，拥有独立 SN 号（QR 识别结果）。
        一个 SN 号对应一个文件夹，该点位照片与 XML 一同保存。
        未识别到 SN 号时，统一保存到「未识别到SN号」文件夹。

        Args:
            scheme_name: 方案名称
            sn: 该点位板卡的 SN（QR 识别结果），为空则存到未识别到SN号文件夹
            position_name: 位置名称
            position_index: 位置序号（1-based）
            annotated_image: 标注结果图
            passed: 该点位是否通过（决定照片文件名与 XML 的 test_result）
            timestamp: 时间戳字符串（可选，默认取当前时间）

        Returns:
            Optional[dict]: 保存信息（img_path/xml_path/sn_dir 等）；失败返回 None
        """
        from core.log_manager import log_info, log_error

        date_str, time_str = self._get_date_time_strs()
        if timestamp:
            time_str = timestamp

        # 判断是否识别到 SN
        safe_id = self._sanitize_id(sn) if sn else ""
        if not safe_id:
            # 未识别到 SN：统一保存到「未识别到SN号」文件夹
            board_dir = self._get_unknown_sn_dir(date_str)
            file_id = f"{time_str}"
        else:
            board_dir = self._get_sn_dir(date_str, safe_id)
            file_id = f"{safe_id}_{time_str}"

        os.makedirs(board_dir, exist_ok=True)

        # 判定文本（用于文件名）
        result_text = "OK" if passed else "NG"

        # 保存照片（标注结果图）
        # 注意：cv2.imwrite 不支持中文路径（如「未识别到SN号」文件夹），
        # 因此使用 cv2.imencode + 手动写文件，以支持中文路径。
        img_name = f"{file_id}_pos{position_index}_{position_name}_{result_text}.jpg"
        img_path = os.path.join(board_dir, img_name)
        try:
            ok, buf = cv2.imencode('.jpg', annotated_image,
                                   [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                log_error(f"编码点位照片失败 [{img_path}]")
                return None
            with open(img_path, 'wb') as f:
                f.write(buf.tobytes())
        except Exception as e:  # noqa: BLE001
            log_error(f"保存点位照片失败 [{img_path}]: {e}")
            return None

        # 生成 XML（该点位一条 test 记录）
        # XML 文件名：有 SN 时用 {SN}.xml（每个 SN 一个 XML），未识别 SN 时用 {时间戳}.xml
        xml_file_id = safe_id if safe_id else time_str
        xml_path = self._write_position_xml(board_dir, xml_file_id, sn, passed, img_path)

        # 追加 CSV 日志
        self._append_csv(date_str, {
            '时间': time_str,
            '方案': scheme_name,
            '位置': position_name,
            '产品ID': sn if sn else "未识别到SN号",
            '判定': result_text,
        })

        log_info(f"点位数据已保存: {img_path}")
        return {
            'img_path': img_path,
            'xml_path': xml_path,
            'sn_dir': board_dir,
            'sn': sn,
            'passed': passed,
            'timestamp': time_str,
        }

    def update_position_result(self, sn: str, passed: bool,
                               timestamp: Optional[str] = None) -> Optional[str]:
        """更新某点位（SN）的 XML 判定结果（NG 手工确认后调用）。

        当操作员在 NG 确认中改变某点位的最终判定时，重写该点位的 XML 文件。

        Args:
            sn: 该点位板卡的 SN（QR 识别结果），为空则对应未识别到SN号文件夹
            passed: 更新后的最终判定（OK/NG）
            timestamp: 时间戳字符串（可选，用于定位未识别到SN号时的 XML 文件）

        Returns:
            Optional[str]: 更新后的 XML 路径；失败返回 None
        """
        from core.log_manager import log_info, log_error, log_warning

        date_str, _ = self._get_date_time_strs()
        safe_id = self._sanitize_id(sn) if sn else ""

        if safe_id:
            board_dir = self._get_sn_dir(date_str, safe_id)
            file_id = safe_id
        else:
            board_dir = self._get_unknown_sn_dir(date_str)
            file_id = timestamp or ""

        xml_path = os.path.join(board_dir, f"{file_id}.xml")
        if not os.path.exists(xml_path):
            log_warning(f"更新 XML 失败，文件不存在: {xml_path}")
            return None

        # 读取原 XML 中的 imgurl，仅更新 test_result
        imgurl = ""
        try:
            with open(xml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            import re
            m = re.search(r'imgurl="([^"]*)"', content)
            if m:
                imgurl = m.group(1)
        except Exception as e:  # noqa: BLE001
            log_error(f"读取 XML 失败 [{xml_path}]: {e}")

        test_result = "OK" if passed else "NG"
        xml_content = self._build_xml(sn, test_result, imgurl)
        try:
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            log_info(f"XML 判定已更新: {xml_path} -> {test_result}")
            return xml_path
        except OSError as e:
            log_error(f"写入 XML 失败 [{xml_path}]: {e}")
            return None

    # ── XML 生成 ──

    def _write_position_xml(self, board_dir: str, file_id: str, sn: str,
                            passed: bool, img_path: str) -> Optional[str]:
        """生成点位测试信息 XML（供 MES 上传）。

        格式:
            <?xml version="1.0" encoding="UTF-8"?>
            <tests>
            <test test_sn="**********" test_data="YYYY/MM/DD HH:MM:SS"
                  test_result="OK/NG" imgurl="******"/>
            </tests>

        Args:
            board_dir: 点位目录
            file_id: 文件名标识（SN 或时间戳）
            sn: 原始 SN
            passed: 是否通过
            img_path: 图片绝对路径

        Returns:
            Optional[str]: XML 路径；失败返回 None
        """
        from core.log_manager import log_info, log_error

        now = datetime.now()
        test_date = now.strftime('%Y/%m/%d %H:%M:%S')
        test_result = "OK" if passed else "NG"
        # imgurl 使用绝对路径（后期根据工控机 MES 路径修改）
        imgurl = os.path.abspath(img_path)

        xml_content = self._build_xml(sn, test_result, imgurl)

        xml_path = os.path.join(board_dir, f"{file_id}.xml")
        try:
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            log_info(f"XML 已生成: {xml_path}")
            return xml_path
        except OSError as e:
            log_error(f"写入 XML 失败 [{xml_path}]: {e}")
            return None

    @staticmethod
    def _build_xml(sn: str, test_result: str, imgurl: str) -> str:
        """构建 XML 内容字符串。"""
        test_date = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<tests>\n'
            f'<test test_sn="{sn}" test_data="{test_date}" '
            f'test_result="{test_result}" imgurl="{imgurl}">\n'
            '</test>\n'
            '</tests>\n'
        )

    # ── 路径辅助 ──

    @staticmethod
    def _get_date_time_strs():
        now = datetime.now()
        return now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S.%f')[:15]

    @staticmethod
    def _sanitize_id(raw_id: str) -> str:
        """将 ID 中的不安全字符替换为下划线"""
        if not raw_id:
            return ""
        return raw_id.replace("/", "_").replace("\\", "_").replace(" ", "_")

    def _get_date_dir(self, date_str: str) -> str:
        """获取某天的根目录: production data/YYYY-MM-DD/"""
        return os.path.join(self._production_dir, date_str)

    def _get_sn_dir(self, date_str: str, safe_id: str) -> str:
        """获取某 SN 的点位目录: production data/YYYY-MM-DD/{SN}/"""
        return os.path.join(self._production_dir, date_str, safe_id)

    def _get_unknown_sn_dir(self, date_str: str) -> str:
        """获取未识别到 SN 号的目录: production data/YYYY-MM-DD/未识别到SN号/"""
        return os.path.join(self._production_dir, date_str, UNKNOWN_SN_DIR)

    # ── CSV 日志 ──

    def _append_csv(self, date_str: str, data: dict):
        """追加日志到 CSV（OK/NG 统一记录在一个 csv 中）"""
        from core.log_manager import log_info, log_error
        csv_path = os.path.join(self._production_dir, date_str, 'test_log.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.exists(csv_path)
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=list(data.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)
            log_info(f"CSV 已追加: {csv_path}")
        except Exception as e:  # noqa: BLE001
            log_error(f"写入 CSV 失败 [{csv_path}]: {e}")

    # ── 旧数据清理 ──

    def clean_old_data(self, retention_days: int = 90):
        """清理 retention_days 天前的生产数据

        Args:
            retention_days: 保留天数（默认 90 天）
        """
        cutoff = datetime.now() - timedelta(days=retention_days)
        if not os.path.exists(self._production_dir):
            return
        for dir_name in os.listdir(self._production_dir):
            dir_path = os.path.join(self._production_dir, dir_name)
            if not os.path.isdir(dir_path):
                continue
            try:
                dir_date = datetime.strptime(dir_name, '%Y-%m-%d')
                if dir_date < cutoff:
                    import shutil
                    shutil.rmtree(dir_path)
            except ValueError:
                continue
