# -*- coding: utf-8 -*-

import csv
import os
from datetime import datetime, timedelta
from typing import List, Optional

import cv2
import numpy as np

from core.paths import PRODUCTION_DATA_DIR


class ResultStorage:
    """生产数据存储管理器

    目录结构:
        data/production data/
            YYYY-MM-DD/
                OK/
                    ok_log.csv              # 当天 OK 的 CSV 日志
                    {ID号}/
                        {ID号}_{HHMMSS}_thumbnail.jpg  # OK 缩略图
                NG/
                    ng_log.csv              # 当天 NG 的 CSV 日志
                    {ID号}/
                        {ID号}_{HHMMSS}_result.jpg     # NG 标注结果图
    """

    def __init__(self):
        self._production_dir = PRODUCTION_DATA_DIR
        os.makedirs(self._production_dir, exist_ok=True)

    # ── 公共保存接口 ──

    def save_ok_data(self, scheme_name: str, product_id: str,
                     annotated_image: np.ndarray):
        """保存 OK 数据：缩略图 + CSV 日志

        Args:
            scheme_name: 方案名称
            product_id: 产品 ID（一维码数据）
            annotated_image: 标注结果图（将生成缩略图保存）
        """
        date_str, time_str = self._get_date_time_strs()
        safe_id = self._sanitize_id(product_id)

        # 目录: production data/YYYY-MM-DD/OK/{ID号}/
        ok_dir = self._get_ok_dir(date_str, safe_id)
        os.makedirs(ok_dir, exist_ok=True)

        # 保存缩略图（最长边 800px）
        thumb_path = os.path.join(ok_dir, f"{safe_id}_{time_str}_thumbnail.jpg")
        self._save_thumbnail(annotated_image, thumb_path, max_size=800)

        # 追加 CSV 日志
        self._append_ok_csv(date_str, {
            '时间': time_str,
            '方案': scheme_name,
            '产品ID': product_id,
            '判定': 'OK',
        })

    def save_ng_data(self, scheme_name: str, product_id: str,
                     annotated_image: np.ndarray):
        """保存 NG 数据：标注结果图 + CSV 日志

        Args:
            scheme_name: 方案名称
            product_id: 产品 ID（一维码数据）
            annotated_image: 标注结果图
        """
        date_str, time_str = self._get_date_time_strs()
        safe_id = self._sanitize_id(product_id)

        # 目录: production data/YYYY-MM-DD/NG/{ID号}/
        ng_dir = self._get_ng_dir(date_str, safe_id)
        os.makedirs(ng_dir, exist_ok=True)

        # 保存标注结果图
        result_path = os.path.join(ng_dir, f"{safe_id}_{time_str}_result.jpg")
        cv2.imwrite(result_path, annotated_image)

        # 追加 CSV 日志
        self._append_ng_csv(date_str, {
            '时间': time_str,
            '方案': scheme_name,
            '产品ID': product_id,
            '判定': 'NG',
        })

    def save_board_data(self, scheme_name: str, sn: str,
                        annotated_image: np.ndarray, passed: bool,
                        save_thumbnail: bool = True) -> Optional[str]:
        """保存单个板卡（点位）的检测数据，并按 SN 生成 XML（供 MES 上传）。

        目录结构:
            data/production data/
                YYYY-MM-DD/
                    OK/ 或 NG/
                        {SN}/
                            {SN}_{HHMMSS}_thumbnail.jpg   # OK 缩略图
                            {SN}_{HHMMSS}_result.jpg      # NG 标注结果图
                            {SN}.xml                       # MES 上传用 XML

        Args:
            scheme_name: 方案名称
            sn: 板卡 SN（QR 识别结果），作为目录/文件名 ID
            annotated_image: 标注结果图
            passed: 该板卡是否通过（决定存 OK 还是 NG 目录）
            save_thumbnail: 是否保存缩略图（OK 时保存缩略图，NG 时保存原图）

        Returns:
            Optional[str]: 保存的图片绝对路径；失败返回 None
        """
        from core.log_manager import log_info, log_error

        date_str, time_str = self._get_date_time_strs()
        safe_id = self._sanitize_id(sn)

        if passed:
            board_dir = self._get_ok_dir(date_str, safe_id)
        else:
            board_dir = self._get_ng_dir(date_str, safe_id)
        os.makedirs(board_dir, exist_ok=True)

        # 保存图片
        if passed and save_thumbnail:
            img_path = os.path.join(board_dir, f"{safe_id}_{time_str}_thumbnail.jpg")
            self._save_thumbnail(annotated_image, img_path, max_size=800)
        else:
            img_path = os.path.join(board_dir, f"{safe_id}_{time_str}_result.jpg")
            cv2.imwrite(img_path, annotated_image)

        # 生成 XML（供 MES 上传）
        self._write_board_xml(board_dir, safe_id, sn, passed, img_path)

        # 追加 CSV 日志
        if passed:
            self._append_ok_csv(date_str, {
                '时间': time_str,
                '方案': scheme_name,
                '产品ID': sn,
                '判定': 'OK',
            })
        else:
            self._append_ng_csv(date_str, {
                '时间': time_str,
                '方案': scheme_name,
                '产品ID': sn,
                '判定': 'NG',
            })

        log_info(f"板卡数据已保存: {img_path}")
        return img_path

    def _write_board_xml(self, board_dir: str, safe_id: str, sn: str,
                         passed: bool, img_path: str):
        """生成板卡测试信息 XML（供 MES 上传）。

        格式:
            <test test_sn="**********" test_date="YYYY/MM/DD HH:MM:SS"
                  test_result="OK/NG" imgurl="******"/>

        Args:
            board_dir: 板卡目录
            safe_id: 安全化后的 SN（用于文件名）
            sn: 原始 SN
            passed: 是否通过
            img_path: 图片绝对路径
        """
        from core.log_manager import log_info, log_error

        now = datetime.now()
        test_date = now.strftime('%Y/%m/%d %H:%M:%S')
        test_result = "OK" if passed else "NG"
        # imgurl 使用绝对路径
        imgurl = os.path.abspath(img_path)

        xml_content = (
            f'<test test_sn="{sn}" test_date="{test_date}" '
            f'test_result="{test_result}" imgurl="{imgurl}"/>'
        )

        xml_path = os.path.join(board_dir, f"{safe_id}.xml")
        try:
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            log_info(f"XML 已生成: {xml_path}")
        except OSError as e:
            log_error(f"写入 XML 失败 [{xml_path}]: {e}")

    # ── 路径辅助 ──

    @staticmethod
    def _get_date_time_strs():
        now = datetime.now()
        return now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S.%f')[:15]

    @staticmethod
    def _sanitize_id(raw_id: str) -> str:
        """将 ID 中的不安全字符替换为下划线"""
        if not raw_id:
            return "UNKNOWN"
        return raw_id.replace("/", "_").replace("\\", "_").replace(" ", "_")

    def _get_date_dir(self, date_str: str) -> str:
        """获取某天的根目录: production data/YYYY-MM-DD/"""
        return os.path.join(self._production_dir, date_str)

    def _get_ok_dir(self, date_str: str, safe_id: str) -> str:
        """获取 OK 的 ID 子目录: production data/YYYY-MM-DD/OK/{ID号}/"""
        return os.path.join(self._production_dir, date_str, 'OK', safe_id)

    def _get_ng_dir(self, date_str: str, safe_id: str) -> str:
        """获取 NG 的 ID 子目录: production data/YYYY-MM-DD/NG/{ID号}/"""
        return os.path.join(self._production_dir, date_str, 'NG', safe_id)

    # ── CSV 日志 ──

    def _append_ok_csv(self, date_str: str, data: dict):
        """追加 OK 日志到 CSV"""
        from core.log_manager import log_info
        csv_path = os.path.join(self._production_dir, date_str, 'OK', 'ok_log.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.exists(csv_path)
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=list(data.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)
            log_info(f"OK CSV 已追加: {csv_path}")
        except Exception as e:
            from core.log_manager import log_error
            log_error(f"写入 OK CSV 失败 [{csv_path}]: {e}")

    def _append_ng_csv(self, date_str: str, data: dict):
        """追加 NG 日志到 CSV"""
        from core.log_manager import log_info
        csv_path = os.path.join(self._production_dir, date_str, 'NG', 'ng_log.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.exists(csv_path)
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=list(data.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)
            log_info(f"NG CSV 已追加: {csv_path}")
        except Exception as e:
            from core.log_manager import log_error
            log_error(f"写入 NG CSV 失败 [{csv_path}]: {e}")

    # ── 缩略图 ──

    @staticmethod
    def _save_thumbnail(image: np.ndarray, save_path: str, max_size: int = 800):
        """保存缩略图，保持宽高比，最长边不超过 max_size

        Args:
            image: 输入图像
            save_path: 保存路径
            max_size: 最长边的最大像素值
        """
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            thumb = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            thumb = image
        cv2.imwrite(save_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

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
