#!/usr/bin/env python
"""开放词汇检测调试脚本（实验 2）。

用途：把图片 + 活动配置喂给 YOLO-World，输出每个物品的 bbox，
保存标注图和 JSON，并打印置信度分布，帮你决定阈值定多少。

用法：
  # 用活动配置里的 objects
  python detect.py --config activities/configs/paper_cup.yaml 图片1.jpg 图片2.jpg

  # 临时指定要找的东西（不用改配置文件）
  python detect.py --prompt "paper cup" "scissors" 图片.jpg

  # 小物体（剪刀、笔）建议提高分辨率
  python detect.py --config ... --imgsz 960 图片.jpg

  # 只看某一类的低分候选，用来标定阈值
  python detect.py --config ... --min-conf 0.01 图片.jpg

输出：
  out/<图片名>_detected.jpg   带框的图
  out/<图片名>_detected.json  结构化结果（可直接喂给后续模块）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 允许直接 `python detect.py` 运行（把项目根加入 sys.path）
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import yaml  # noqa: E402

from vision.object_detector import (  # noqa: E402
    DetectionResult,
    ObjectDetector,
    ObjectSpec,
    specs_from_config,
)


def build_specs(args: argparse.Namespace) -> list[ObjectSpec]:
    if args.prompt:
        return [ObjectDetector._spec_from_str(p) for p in args.prompt]
    if not args.config:
        sys.exit("必须提供 --config <yaml> 或 --prompt <英文短语...>")
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"配置文件不存在: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    body = cfg.get("activity", cfg)
    specs = specs_from_config(body)
    if not specs:
        sys.exit(f"配置里没有可用的 objects: {cfg_path}")
    # 命令行统一覆盖阈值
    if args.min_conf is not None:
        for sp in specs:
            sp.conf = args.min_conf
    return specs


def read_image(path: Path):
    """cv2.imread 不支持中文路径，改用 imdecode。"""
    data = path.read_bytes()
    import numpy as np

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"不是有效的图片文件: {path}")
    return img


def report(res: DetectionResult, specs: list[ObjectSpec], top_n: int = 5) -> None:
    w, h = res.image_size
    print(f"\n=== {Path(res.source).name}  ({w}x{h})  "
          f"device={res.device}  推理 {res.infer_ms:.0f} ms ===")
    if not res.detections:
        print("  ⚠ 没检测到任何目标。试试：--min-conf 0.01 / --imgsz 960 / 换 prompt 说法")
        return
    for sp in specs:
        items = res.by_object(sp.id)
        if not items:
            print(f"  {sp.name_cn:<6} ({sp.id})  — 未检出   阈值={sp.conf}")
            continue
        confs = [d.confidence for d in items]
        top = items[0]
        x1, y1, x2, y2 = top.bbox_xyxy
        # 多实例时给分布摘要，而不是逐个刷屏
        stat = (f"最高={max(confs):.3f} 中位={sorted(confs)[len(confs) // 2]:.3f} "
                f"最低={min(confs):.3f}")
        print(f"  {sp.name_cn:<6} ({sp.id})  {len(items):>3} 个  {stat}")
        print(f"         最强一个: bbox=[{x1},{y1},{x2},{y2}] 中心={top.center_xy} "
              f"占比={top.area_ratio:.3%} 命中={top.matched_prompt!r}")
        if len(items) > 1:
            print(f"         前 {min(top_n, len(items))} 个分数: "
                  + ", ".join(f"{c:.3f}" for c in confs[:top_n])
                  + (" ..." if len(items) > top_n else ""))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="YOLO-World 开放词汇检测调试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("images", nargs="+", help="一张或多张图片路径")
    ap.add_argument("--config", "-c", help="活动配置 YAML（读 objects 段）")
    ap.add_argument("--prompt", "-p", nargs="+",
                    help='直接指定英文短语，如 --prompt "paper cup" "scissors"')
    ap.add_argument("--size", default="s", choices=list("nsmlx"),
                    help="模型大小，默认 s（速度/精度平衡，适合 M 系列 Mac）")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="推理分辨率，小物体建议 960 或 1280，默认 640")
    ap.add_argument("--device", default="auto", help="auto / mps / cpu / cuda")
    ap.add_argument("--min-conf", type=float, default=None,
                    help="统一覆盖所有物体的置信度阈值（标定阈值时用 0.01）")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    ap.add_argument("--dup-iou", type=float, default=0.55,
                    help="跨 prompt 去重阈值。同一物体的两个框 IoU 超过此值视为同一实例，"
                         "只留高分。堆叠纸杯这种紧密排列建议 0.6~0.75，默认 0.55")
    ap.add_argument("--max-det", type=int, default=100,
                    help="单个物体最多保留几个实例，防止低阈值下爆量，默认 100")
    ap.add_argument("--out", default=str(ROOT / "out"), help="输出目录")
    ap.add_argument("--no-image", action="store_true", help="只输出 JSON，不存标注图")
    args = ap.parse_args()

    specs = build_specs(args)
    print("待检测物品：")
    for sp in specs:
        print(f"  - {sp.name_cn} ({sp.id}) conf>={sp.conf} prompts={sp.prompts}")

    t0 = time.time()
    det = ObjectDetector(specs, model_size=args.size, device=args.device,
                         iou=args.iou, imgsz=args.imgsz,
                         dup_iou=args.dup_iou, max_det=args.max_det,
                         verbose=True)
    print(f"模型加载耗时 {time.time() - t0:.1f}s\n")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_infer = 0.0
    for raw in args.images:
        path = Path(raw).expanduser()
        if not path.exists():
            print(f"⚠ 跳过（文件不存在）: {path}")
            continue
        try:
            frame = read_image(path)
        except ValueError as e:
            print(f"⚠ {e}")
            continue

        # 传 ndarray，绕开 OpenCV 的中文路径问题；source 记录原路径
        res = det.detect(frame)
        res.source = str(path)
        total_infer += res.infer_ms
        report(res, specs)

        if not args.no_image:
            annotated = det.draw(res, frame)
            img_path = out_dir / f"{path.stem}_detected.jpg"
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                buf.tofile(str(img_path))
            else:
                img_path = None
        else:
            img_path = None

        import json

        json_path = out_dir / f"{path.stem}_detected.json"
        json_path.write_text(
            json.dumps(res.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if img_path:
            print(f"  → 标注图: {img_path}")
        print(f"  → JSON:  {json_path}")

    n = len(args.images) or 1
    print(f"\n平均每张推理 {total_infer / n:.0f} ms  "
          f"(约 {1000 / max(total_infer / n, 1e-6):.1f} FPS，含预处理见下方说明)")
    print("提示：单张推理时间 ≠ 视频流帧率。接视频时检测跑 2~5 FPS 就够，"
          "中间帧交给 tracking 补。")


if __name__ == "__main__":
    main()
