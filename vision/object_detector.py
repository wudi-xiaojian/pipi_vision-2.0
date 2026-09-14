"""开放词汇目标检测模块（YOLO-World）。

职责边界：只回答"画面里有哪些活动材料、分别在哪"，
不判断动作、不判断活动步骤（那是 Activity Engine 的事）。

设计要点：
1. set_classes() 只在活动切换时调用一次。该调用会重算 CLIP 文本
   embedding，逐帧调用会把帧率打死。
2. 每个物体支持多个英文 prompt，取最高分。YOLO-World 的文本编码器
   是 CLIP，对英文短语效果显著好于中文，因此配置里 name_cn 只用于
   展示，实际喂给模型的是 prompts。
3. 开放词汇检测的置信度天然偏低，默认阈值要比普通 YOLO 低得多。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


@dataclass
class ObjectSpec:
    """一个待检测物体的定义。"""

    id: str
    name_cn: str
    prompts: list[str]
    conf: float = 0.05
    color: tuple[int, int, int] = field(default=(0, 200, 0))  # BGR


@dataclass
class Detection:
    """单个检测结果。"""

    object_id: str
    name_cn: str
    matched_prompt: str
    confidence: float
    bbox_xyxy: list[int]
    center_xy: list[int]
    area_ratio: float  # bbox 面积 / 图像面积，用于粗判远近


@dataclass
class DetectionResult:
    """一帧/一张图的完整检测结果。"""

    source: str
    image_size: list[int]  # [w, h]
    device: str
    infer_ms: float
    detections: list[Detection]

    def by_object(self, object_id: str) -> list[Detection]:
        return [d for d in self.detections if d.object_id == object_id]

    def top(self, object_id: str) -> Detection | None:
        """某个物体里置信度最高的那个。"""
        items = self.by_object(object_id)
        return max(items, key=lambda d: d.confidence) if items else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "image_size": self.image_size,
            "device": self.device,
            "infer_ms": round(self.infer_ms, 1),
            "objects": [asdict(d) for d in self.detections],
        }


# 默认调色板（BGR），按 ObjectSpec 顺序循环取用
_PALETTE: list[tuple[int, int, int]] = [
    (0, 200, 0),
    (0, 140, 255),
    (255, 120, 0),
    (0, 0, 255),
    (200, 0, 200),
    (0, 220, 220),
    (255, 0, 120),
    (180, 180, 0),
]


class ObjectDetector:
    """YOLO-World 开放词汇检测器封装。"""

    def __init__(
        self,
        specs: Sequence[ObjectSpec] | Sequence[str],
        model_size: str = "s",
        device: str = "auto",
        iou: float = 0.45,
        imgsz: int = 640,
        dup_iou: float = 0.55,
        max_det: int = 100,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            specs: ObjectSpec 列表，或直接给英文短语字符串列表（自动包装）。
            model_size: yolov8s-world / yolov8m-world 等的尺寸后缀，
                        可选 n/s/m/l/x。M 系列 Mac 建议 s 或 m。
            device: 'auto' 会优先选 Apple Silicon 的 mps，失败则回退 cpu。
            iou: NMS 的 IoU 阈值。
            imgsz: 推理输入分辨率。小物体（如剪刀、笔）建议提到 960/1280。
            dup_iou: 跨 prompt 去重阈值。同一物体内两个框 IoU 超过此值，
                     视为同一物理实例，只保留高分那个。
                     调低 → 去重更狠（可能把挨得很近的两个杯子并成一个）；
                     调高 → 保留更多（可能出现同一杯子重复计数）。
                     堆叠纸杯这种紧密排列场景建议 0.6~0.75。
            max_det: 单个物体最多保留几个实例，防止低阈值下爆量。
            verbose: 是否打印 ultralytics 内部日志。
        """
        from ultralytics import YOLO

        self.specs: list[ObjectSpec] = [
            s if isinstance(s, ObjectSpec) else self._spec_from_str(s)
            for s in specs
        ]
        for i, sp in enumerate(self.specs):
            if sp.color == (0, 200, 0) and i < len(_PALETTE):
                sp.color = _PALETTE[i % len(_PALETTE)]

        self.model_size = model_size
        self.iou = iou
        self.imgsz = imgsz
        self.dup_iou = dup_iou
        self.max_det = max_det
        self.device = self._resolve_device(device)

        self.model = YOLO(f"yolov8{model_size}-world.pt")
        # 关键：只在初始化时设置一次类别
        self.model.set_classes(self._flat_prompts())
        self._prompt_owner = self._build_prompt_owner()

        if verbose:
            print(f"[ObjectDetector] device={self.device} imgsz={imgsz} "
                  f"classes={len(self._flat_prompts())} "
                  f"objects={len(self.specs)}")

    # ---------- 构造辅助 ----------

    @staticmethod
    def _spec_from_str(text: str) -> ObjectSpec:
        slug = text.strip().lower().replace(" ", "_")
        return ObjectSpec(id=slug, name_cn=text.strip(), prompts=[text.strip()])

    def _flat_prompts(self) -> list[str]:
        """把所有 spec 的 prompts 摊平成一维列表，顺序即类别索引。"""
        out: list[str] = []
        for sp in self.specs:
            out.extend(sp.prompts)
        return out

    def _build_prompt_owner(self) -> list[int]:
        """类别索引 -> 它属于第几个 ObjectSpec。"""
        owner: list[int] = []
        for i, sp in enumerate(self.specs):
            owner.extend([i] * len(sp.prompts))
        return owner

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    # ---------- 主要接口 ----------

    def detect(self, image: str | Path | np.ndarray) -> DetectionResult:
        """检测一张图。image 可以是路径或已读入的 BGR ndarray。"""
        if isinstance(image, np.ndarray):
            frame = image
            source = "<ndarray>"
        else:
            path = Path(image)
            frame = cv2.imread(str(path))
            if frame is None:
                raise FileNotFoundError(f"无法读取图片: {path}")
            source = str(path)

        h, w = frame.shape[:2]

        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self._min_conf(),
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        r = results[0]
        infer_ms = float(r.speed.get("inference", 0.0)) if r.speed else 0.0

        detections = self._parse(r, w, h)
        return DetectionResult(
            source=source,
            image_size=[w, h],
            device=self.device,
            infer_ms=infer_ms,
            detections=detections,
        )

    def _min_conf(self) -> float:
        return min(sp.conf for sp in self.specs)

    @staticmethod
    def _iou(a: Sequence[float], b: Sequence[float]) -> float:
        """两个 xyxy 框的 IoU。"""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _parse(self, r: Any, w: int, h: int) -> list[Detection]:
        """解析检测结果：保留每个物理实例，跨 prompt 去重。

        为什么不按 spec 只取最高分：
          纸杯搭建这类活动，"有几个杯子""怎么堆的"本身就是核心信息。
          每类只留一个框会让下游状态机彻底失明。

        为什么要跨 prompt 去重：
          同一个杯子可能同时被 "paper cup" 和 "stacked paper cups" 命中，
          ultralytics 默认按类别做 NMS（非 agnostic），两个框都会留下。
          不去重就会把一个杯子数成两个。
          做法：同一 spec 内按置信度降序，IoU 超过阈值的视为同一实例，只留最高分。
        """
        img_area = float(w * h) or 1.0

        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = r.names  # {类别索引: prompt 文本}
        # spec_idx -> [(conf, cls_id, xyxy), ...]，保留全部候选
        grouped: dict[int, list[tuple[float, int, list[float]]]] = {}

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            if cls_id >= len(self._prompt_owner):
                continue
            spec_idx = self._prompt_owner[cls_id]
            spec = self.specs[spec_idx]
            if conf < spec.conf:
                continue
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            grouped.setdefault(spec_idx, []).append((conf, cls_id, xyxy))

        out: list[Detection] = []
        for spec_idx, cands in grouped.items():
            spec = self.specs[spec_idx]
            # 高分优先，低分重叠框被丢弃
            cands.sort(key=lambda t: -t[0])
            kept: list[tuple[float, int, list[float]]] = []
            for conf, cls_id, xyxy in cands:
                if any(self._iou(xyxy, k[2]) >= self.dup_iou for k in kept):
                    continue  # 与已保留的更高框指向同一物理实例
                kept.append((conf, cls_id, xyxy))

            for conf, cls_id, xyxy in kept[: self.max_det]:
                x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                out.append(
                    Detection(
                        object_id=spec.id,
                        name_cn=spec.name_cn,
                        matched_prompt=str(names.get(cls_id, spec.prompts[0])),
                        confidence=round(conf, 4),
                        bbox_xyxy=[x1, y1, x2, y2],
                        center_xy=[(x1 + x2) // 2, (y1 + y2) // 2],
                        area_ratio=round((x2 - x1) * (y2 - y1) / img_area, 5),
                    )
                )
        out.sort(key=lambda d: -d.confidence)
        return out

    # ---------- 可视化 ----------

    def draw(self, result: DetectionResult, frame: np.ndarray | None = None,
             show_conf: bool = True) -> np.ndarray:
        """把检测结果画到图上，返回新的 BGR ndarray（不改原图）。"""
        if frame is None:
            frame = cv2.imread(result.source)
            if frame is None:
                raise FileNotFoundError(f"无法读取图片: {result.source}")
        canvas = frame.copy()
        color_of = {sp.id: sp.color for sp in self.specs}
        # 每个物体独立编号，方便肉眼核对计数
        counter: dict[str, int] = {}

        for d in result.detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            color = color_of.get(d.object_id, (0, 200, 0))
            counter[d.object_id] = counter.get(d.object_id, 0) + 1
            idx = counter[d.object_id]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            label = d.name_cn if d.name_cn else d.object_id
            label = f"{label}#{idx}"
            if show_conf:
                label = f"{label} {d.confidence:.2f}"
            # 文本框
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = max(y1 - th - baseline - 2, 0)
            cv2.rectangle(canvas, (x1, ty), (x1 + tw + 6, ty + th + baseline + 4), color, -1)
            cv2.putText(canvas, label, (x1 + 3, ty + th + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            # 中心点
            cx, cy = d.center_xy
            cv2.circle(canvas, (cx, cy), 3, color, -1)

        return canvas

    def save(self, result: DetectionResult, out_dir: str | Path,
             stem: str | None = None) -> tuple[Path, Path]:
        """保存标注图和 JSON，返回 (图片路径, JSON 路径)。"""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = stem or Path(result.source).stem

        img_path = out_dir / f"{stem}_detected.jpg"
        json_path = out_dir / f"{stem}_detected.json"

        frame = cv2.imread(result.source)
        annotated = self.draw(result, frame)
        # OpenCV 写中文路径不稳，用 imencode + tofile
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("图片编码失败")
        buf.tofile(str(img_path))

        json_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return img_path, json_path


def specs_from_config(cfg: dict) -> list[ObjectSpec]:
    """从活动配置 dict 里读出 objects 段，转成 ObjectSpec 列表。"""
    raw = cfg.get("objects") or []
    specs: list[ObjectSpec] = []
    for item in raw:
        if isinstance(item, str):
            specs.append(ObjectDetector._spec_from_str(item))
            continue
        prompts = item.get("prompts") or ([item["name"]] if item.get("name") else [])
        if not prompts:
            continue
        specs.append(
            ObjectSpec(
                id=str(item.get("id") or prompts[0]).replace(" ", "_"),
                name_cn=str(item.get("name_cn") or item.get("name") or prompts[0]),
                prompts=list(prompts),
                conf=float(item.get("conf", 0.05)),
                color=tuple(item.get("color", (0, 200, 0))),
            )
        )
    return specs
