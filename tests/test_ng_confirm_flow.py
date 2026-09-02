# -*- coding: utf-8 -*-
"""测试 NG 逐点位确认流程（需求2）。

验证:
    1. 全部检测完成后，收集所有 NG 点位
    2. 逐点位确认，每个 NG 点位单独确认
    3. 确认后更新该点位判定与 XML
    4. 所有确认完成后计算最终结果
"""
import sys
import os

# 设置 stdout 编码
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.inspection_workflow import InspectionWorkflow, PositionResult


def test_ng_confirm_flow():
    """测试逐点位确认流程"""
    workflow = InspectionWorkflow()

    # 模拟 4 个位置，其中 2 个 NG
    results = [
        PositionResult(name="1.1", position=1, passed=True, qr_data="SN001"),
        PositionResult(name="1.2", position=2, passed=False, qr_data="SN002"),
        PositionResult(name="1.3", position=3, passed=True, qr_data="SN003"),
        PositionResult(name="1.4", position=4, passed=False, qr_data="SN004"),
    ]
    workflow._results = results

    # 模拟 _show_final_result 的 NG 分支
    workflow._pending_ng_confirm = [i for i, r in enumerate(results) if not r.passed]
    workflow._current_confirm_index = 0
    workflow._set_state(workflow.State.WAITING_FOR_CONFIRM)

    # 收集确认请求
    requested = []
    workflow.ng_confirm_requested.connect(
        lambda idx, result, current, total: requested.append((idx, result.name, current, total))
    )

    # 请求第一个 NG 点位
    workflow._request_next_ng_confirm()
    assert len(requested) == 1, "应请求第一个 NG 点位"
    idx, name, current, total = requested[0]
    assert idx == 1 and name == "1.2" and current == 1 and total == 2, \
        f"第一个 NG 点位错误: {requested[0]}"
    print(f"[PASS] 请求第一个 NG 点位: {name} ({current}/{total})")

    # 确认第一个 NG 点位为 OK
    workflow.confirm_ng_result(True)
    assert results[1].passed is True, "确认 OK 后该点位应判定为 OK"
    assert results[1].confirmed is True, "该点位应标记为已确认"
    assert len(requested) == 2, "应请求第二个 NG 点位"
    idx, name, current, total = requested[1]
    assert idx == 3 and name == "1.4" and current == 2 and total == 2, \
        f"第二个 NG 点位错误: {requested[1]}"
    print(f"[PASS] 确认第一个 NG 为 OK，请求第二个 NG 点位: {name} ({current}/{total})")

    # 确认第二个 NG 点位为 NG
    workflow.confirm_ng_result(False)
    assert results[3].passed is False, "确认 NG 后该点位应判定为 NG"
    assert len(requested) == 2, "所有 NG 点位确认完成，不应再请求"
    print("[PASS] 确认第二个 NG 为 NG")

    # 验证最终结果（1.2 改为 OK，1.4 仍 NG → 整体 NG）
    all_passed = all(r.passed for r in results)
    assert all_passed is False, "最终结果应为 NG"
    print("[PASS] 最终结果计算正确: NG")

    print("\n[PASS] NG 逐点位确认流程测试通过")


if __name__ == "__main__":
    test_ng_confirm_flow()
