#!/usr/bin/env python
"""追踪调试脚本（实验 3）。

视频 / 摄像头 → YOLO-World 检测 + 身份维持追踪，
输出带 track_id 的标注视频和逐帧 JSONL（Activity Engine 的输入格式）。

用法：
  # 视频文件（自动读取 fps，可 --fps 覆盖）
  python track.py --video 杯子搭建.mp4 --config activities/configs/paper_cup.yaml

  # 自己的摄像头实时跑
  python track.py --camera 0 --config activities/configs/paper_cup.yaml

  # 检测降频：每 3 帧跑一次 YOLO-World，中间帧纯靠预测滑行
  python track.py --video in.mp4 --config ... --detect-every 3

  # 遮挡容忍调大（默认 75 帧，30fps 下约 2.5 秒）
  python track.py --video in.mp4 --max-coast 90

输出：
  out/<视频名>_tracked.mp4    带 ID、轨迹尾巴、预测态虚线的视频
  out/<视频名>_tracks.jsonl   每行一帧的结构化轨迹（给状态机喂的）
  结尾打印统计：各物品确认轨迹数、最长滑行帧数、检测频率与平均耗时
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from vision.object_tracker import ObjectTracker  # noqa: E402
from vision.object_detector import ObjectSpec, specs_from_config  # noqa: E402

try:
    import yaml  # noqa: E402
except ImportError:  # pragma: no cover
    yaml = None


def build_specs(args: argparse.Namespace) -> list[ObjectSpec]:
    if args.prompt:
        return [ObjectSpec(id=p.replace(" ", "_"), name_cn=p, prompts=[p],
                           conf=0.02) for p in args.prompt]
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
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO-World 检测 + 多目标追踪调试")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="视频文件路径")
    src.add_argument("--camera", type=int, help="摄像头序号（Mac 一般为 0）")
    ap.add_argument("--config", "-c", help="活动配置 YAML")
    ap.add_argument("--prompt", "-p", nargs="+", help="直接指定英文短语")
    ap.add_argument("--size", default="s", choices=list("nsmlx"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-conf", type=float, default=None,
                    help="统一覆盖所有物品的置信度阈值（追踪建议 0.15~0.2）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dup-iou", type=float, default=0.65)
    ap.add_argument("--detect-every", type=int, default=1,
                    help="每 N 帧跑一次检测，其余帧预测滑行。默认 1")
    ap.add_argument("--match-iou", type=float, default=0.25,
                    help="关联 IoU 阈值，密集堆叠可降到 0.15")
    ap.add_argument("--max-coast", type=int, default=75,
                    help="遮挡容忍帧数（fps=30 时 75 帧约 2.5 秒）")
    ap.add_argument("--min-hits", type=int, default=3,
                    help="观测几次后转 confirmed，默认 3")
    ap.add_argument("--fps", type=float, default=None,
                    help="覆盖视频 fps（默认读文件属性）")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="只处理前 N 帧（快速验证用）")
    ap.add_argument("--show", action="store_true",
                    help="视频模式也开预览窗口（摄像头模式默认开）")
    ap.add_argument("--no-show", action="store_true",
                    help="摄像头模式关闭预览窗口（纯后台录制）")
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--no-video", action="store_true", help="不输出标注视频")
    args = ap.parse_args()

    specs = build_specs(args)

    # 统一覆盖阈值（标定用；低阈值会带入"半截框"类低质检出）
    if args.min_conf is not None:
        for sp in specs:
            sp.conf = args.min_conf

    cap = (cv2.VideoCapture(args.camera) if args.camera is not None
           else cv2.VideoCapture(args.video))
    if not cap.isOpened():
        sys.exit(f"打不开视频源: {args.video or args.camera}")

    # 摄像头必须先读一帧才能拿到真实分辨率，否则 W/H 是 0，
    # VideoWriter 会写出损坏文件。
    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        sys.exit("读不到第一帧。摄像头模式请确认：\n"
                 "  1) 已授予终端「摄像头」权限（系统设置 > 隐私与安全性 > 摄像头）\n"
                 "  2) 没有其他程序（会议/相机）正在占用摄像头\n"
                 "  3) --camera 序号正确，Mac 内置摄像头通常是 0")
    H, W = first.shape[:2]
    vfps = cap.get(cv2.CAP_PROP_FPS)
    fps = args.fps or (vfps if vfps and vfps > 1 else 30.0)

    is_camera = args.camera is not None
    # 摄像头模式默认开预览（否则看不到任何画面，像卡死）
    show = args.show or (is_camera and not args.no_show)

    tracker = ObjectTracker(
        specs,
        model_size=args.size,
        device=args.device,
        imgsz=args.imgsz,
        dup_iou=args.dup_iou,
        detect_every=args.detect_every,
        match_iou=args.match_iou,
        max_coast_frames=args.max_coast,
        min_hits=args.min_hits,
        fps=fps,
        verbose=True,
    )

    stem = f"camera_{time.strftime('%H%M%S')}" if is_camera else Path(args.video).stem
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{stem}_tracks.jsonl"
    video_path = out_dir / f"{stem}_tracked.mp4"

    writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (W, H))
        if not writer.isOpened():
            print(f"⚠ 无法写入 {video_path}，本次只输出 JSONL")
            writer = None

    print(f"\n视频源: {'摄像头 ' + str(args.camera) if is_camera else args.video}"
          f"  {W}x{H} @ {fps:.1f}fps")
    if show:
        print("预览窗口已打开：按 q 或 Esc 结束（会正常保存结果并打印统计）")
    if is_camera and args.detect_every == 1:
        print("提示：摄像头下检测每帧都跑，MPS 上约 30~60 FPS；"
              "想更流畅可加 --detect-every 3\n")

    t0 = time.time()
    n = 0
    jsonl_f = jsonl_path.open("w", encoding="utf-8")
    interrupted = False

    try:
        while True:
            ok, frame = (True, first) if n == 0 else cap.read()
            first = None  # 首帧已消费
            if not ok or frame is None:
                break
            if args.max_frames and n >= args.max_frames:
                break

            res = tracker.process(frame)
            canvas = tracker.draw(frame, res)
            if writer is not None:
                writer.write(canvas)
            if show:
                cv2.imshow("PiPi Tracking (q/Esc = stop)", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            line = {
                "frame": n,
                "ts": round(n / fps, 3),
                "ran_detection": res.ran_detection,
                "n_detections": len(res.detections),
                "tracks": [t.to_dict() for t in res.tracks],
            }
            jsonl_f.write(json.dumps(line, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0:
                rate = n / (time.time() - t0)
                print(f"  已处理 {n} 帧  实时速率 {rate:.1f} FPS ...")
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到 Ctrl+C，正在保存结果...")
    finally:
        jsonl_f.close()
        cap.release()
        if writer is not None:
            writer.release()  # 必须调用，否则 mp4 不封口无法播放
        if show:
            cv2.destroyAllWindows()

    wall = time.time() - t0
    print(f"\n=== 追踪{'中断' if interrupted else '完成'}：{n} 帧，"
          f"墙钟 {wall:.1f}s，平均 {n / max(wall, 1e-6):.1f} FPS ===")
    if n == 0:
        print("没有处理任何帧，未生成结果。")
        return

    # ---------- 统计 ----------
    print(f"\n=== 追踪完成：{n} 帧，墙钟 {wall:.1f}s，平均 {n / max(wall, 1e-6):.1f} FPS ===")
    print(f"视频源 fps={fps:.1f}  检测频率=每 {args.detect_every} 帧一次 "
          f"(有效 {fps / args.detect_every:.1f} 次/秒)")
    if tracker.infer_times:
        avg = sum(tracker.infer_times) / len(tracker.infer_times)
        print(f"单次检测平均耗时 {avg:.0f} ms (device={tracker.detector.device})")

    per_obj: dict[str, dict] = {}
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        for t in json.loads(line)["tracks"]:
            s = per_obj.setdefault(t["object_id"], {
                "ids": set(), "confirmed": set(), "name": t["name_cn"],
                "max_coast": 0})
            s["ids"].add(t["track_id"])
            if t["confirmed"]:
                s["confirmed"].add(t["track_id"])
            s["max_coast"] = max(s["max_coast"], t["coasting_frames"])

    print("\n物品统计（整段视频累计）：")
    for oid, s in per_obj.items():
        print(f"  {s['name']:<4} ({oid}): 轨迹共 {len(s['ids'])} 个，"
              f"其中确认 {len(s['confirmed'])} 个，"
              f"单轨迹最长滑行 {s['max_coast']} 帧 "
              f"(≈{s['max_coast'] / fps:.1f}s)")
    print(f"\n结果文件：\n  {video_path if not args.no_video else '(未输出视频)'}"
          f"\n  {jsonl_path}")


if __name__ == "__main__":
    main()
