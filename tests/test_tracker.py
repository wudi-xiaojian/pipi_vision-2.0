#!/usr/bin/env python
"""追踪器单元测试：不依赖 YOLO-World，直接验证 SimpleTracker 逻辑。

构造合成场景（移动方块 + 遮挡窗口 + 闪现误检），断言：
  T1 正常移动：ID 全程不变，速度估计接近真值
  T2 遮挡 20 帧：滑行期间保持同一 ID，遮挡结束后回归
  T3 闪现误检（只出现 1 帧）：永远不转 confirmed
  T4 遮挡超过 max_coast：轨迹被正确丢弃
  T5 双目标交叉：各自 ID 不串

用法：python tests/test_tracker.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision.object_detector import Detection  # noqa: E402
from vision.object_tracker import SimpleTracker  # noqa: E402

FPS = 30.0
DT = 1.0 / FPS


def det(object_id: str, name: str, cx: float, cy: float,
        s: float = 60.0, conf: float = 0.5) -> Detection:
    return Detection(
        object_id=object_id, name_cn=name, matched_prompt="x",
        confidence=conf,
        bbox_xyxy=[int(cx - s / 2), int(cy - s / 2),
                   int(cx + s / 2), int(cy + s / 2)],
        center_xy=[int(cx), int(cy)], area_ratio=0.01,
    )


def get_track(tracks, tid):
    return next((t for t in tracks if t.track_id == tid), None)


def run(name: str, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        return False


# ---------------- T1 正常移动 ----------------
def t1():
    tr = SimpleTracker(min_hits=3)
    tid_seen = []
    for i in range(30):
        x = 100 + 5 * i  # 150 px/s 向右
        tracks = tr.step([det("cup", "杯子", x, 200)], i, FPS)
        assert len(tracks) == 1, f"帧{i} 轨迹数 {len(tracks)} != 1"
        tid_seen.append(tracks[0].track_id)
    t = tracks[0]
    assert set(tid_seen) == {tid_seen[0]}, f"ID 发生切换: {set(tid_seen)}"
    assert t.confirmed, "30 帧后仍未 confirmed"
    assert abs(t.vx - 150) < 30, f"速度估计 {t.vx:.0f} 偏离真值 150 太多"
    assert abs(t.vy) < 10, f"y 方向不应有速度: {t.vy:.0f}"


# ---------------- T2 遮挡 20 帧 ----------------
def t2():
    tr = SimpleTracker(min_hits=3, max_coast_frames=75)
    # 先让目标稳定跟踪 10 帧
    for i in range(10):
        tracks = tr.step([det("cup", "杯子", 100 + 5 * i, 200)], i, FPS)
    tid = tracks[0].track_id
    # 遮挡 20 帧：没有检测输入
    for j in range(20):
        i = 10 + j
        tracks = tr.step([], i, FPS)
        t = get_track(tracks, tid)
        assert t is not None, f"滑行第{j}帧 ID {tid} 丢了"
        assert t.is_predicted, f"滑行第{j}帧应处于预测态"
        assert t.coasting_frames == j + 1
    # 遮挡结束时，预测位置应接近真实位置（恒速外推）
    predicted_cx = tracks[0].center[0]
    true_cx = 100 + 5 * 30
    assert abs(predicted_cx - true_cx) < 40, \
        f"外推位置 {predicted_cx:.0f} 偏离真值 {true_cx:.0f}"
    # 目标重新出现，在原轨迹附近 → 应抢回原 ID
    tracks = tr.step([det("cup", "杯子", true_cx, 200)], 30, FPS)
    assert len(tracks) == 1, f"恢复后轨迹数 {len(tracks)}（预期合并为 1）"
    assert tracks[0].track_id == tid, \
        f"ID 不守恒：遮挡前 {tid}，恢复后 {tracks[0].track_id}"
    assert not tracks[0].is_predicted, "恢复后应回到观测态"


# ---------------- T3 闪现误检 ----------------
def t3():
    tr = SimpleTracker(min_hits=3)
    tr.step([det("cup", "杯子", 100, 200)], 0, FPS)
    tracks = tr.step([det("cup", "杯子", 500, 500)], 1, FPS)  # 只闪 1 帧
    flashed = [t for t in tracks if t.center[0] > 400]
    assert len(flashed) == 1
    assert not flashed[0].confirmed, "单帧闪现不应 confirmed"
    # 下一帧闪检消失，它进入滑行，再下一帧被清理掉是正常行为；
    # 但真目标不受影响
    tracks = tr.step([det("cup", "杯子", 110, 200)], 2, FPS)
    assert any(t.center[0] < 200 and t.confidence > 0.4 for t in tracks), \
        "真目标不应被闪检干扰"


# ---------------- T4 遮挡超时丢弃 ----------------
def t4():
    tr = SimpleTracker(min_hits=3, max_coast_frames=15)
    for i in range(5):
        tracks = tr.step([det("cup", "杯子", 100, 200)], i, FPS)
    tid = tracks[0].track_id
    for i in range(5, 25):  # 20 帧 > 15 帧上限
        tracks = tr.step([], i, FPS)
    assert get_track(tracks, tid) is None, "超时轨迹应被丢弃"


# ---------------- T5 双目标交叉 ----------------
def t5():
    tr = SimpleTracker(min_hits=3, match_iou=0.15)
    # A 从左到右，B 从右到左，在第 15 帧交叉
    id_a = id_b = None
    for i in range(31):
        xa = 100 + 12 * i
        xb = 700 - 12 * i
        tracks = tr.step([det("cup", "杯子", xa, 200, conf=0.9),
                          det("cup", "杯子", xb, 201, conf=0.85)], i, FPS)
        if i == 4:
            # 记录初始身份：A 在左（conf 高），B 在右
            left = min(tracks, key=lambda t: t.center[0])
            right = max(tracks, key=lambda t: t.center[0])
            id_a, id_b = left.track_id, right.track_id
        if i >= 25:
            a = get_track(tracks, id_a)
            b = get_track(tracks, id_b)
            assert a and b, f"交叉后轨迹丢失: {len(tracks)} 条"
    # A 现在应该在右侧（走过来了），B 在左侧
    assert a.center[0] > b.center[0], "A/B 身份串了（ID 互换）"


# ---------------- T6 影子轨迹：半截框不养出平行轨迹 ----------------
def t6():
    """demo 实测场景：同一杯子既有整杯框又有下半截框。
    IoU≈0.47 躲过检测去重，containment≈0.85 会持续触发重叠，
    连续 merge_streak 帧后必须被轨迹级合并消灭。"""
    tr = SimpleTracker(min_hits=2, merge_streak=5)
    for i in range(12):
        full = det("cup", "杯子", 300, 500, s=200)          # 整杯
        half = det("cup", "杯子", 300, 555, s=130)          # 下半截
        tracks = tr.step([full, half], i, FPS)
    cups = [t for t in tracks if t.confirmed]
    assert len(cups) == 1, f"同一杯子被拆成 {len(cups)} 条确认轨迹（应合并为 1）"


if __name__ == "__main__":
    print("SimpleTracker 单元测试:")
    results = [
        run("T1 正常移动：ID 稳定 + 速度估计", t1),
        run("T2 遮挡 20 帧：滑行保持 + 原 ID 回归", t2),
        run("T3 闪现误检：不转 confirmed", t3),
        run("T4 遮挡超时：轨迹正确丢弃", t4),
        run("T5 双目标交叉：身份不串", t5),
        run("T6 半截框影子：轨迹级合并", t6),
    ]
    ok = sum(results)
    print(f"\n{ok}/{len(results)} 通过")
    sys.exit(0 if ok == len(results) else 1)
