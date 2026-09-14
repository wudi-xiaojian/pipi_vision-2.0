# YOLO-World 开放词汇检测模块（实验 2）

对应方案文档第 7 节「Open-Vocabulary Object Detection」和第 23 节「实验 2」。

**这一步要回答的问题：不训练专用 YOLO，能不能直接找到活动材料？**

---

## 一、环境准备

本机已就绪（`pipi-vision` conda 环境，Python 3.11.16）：

```bash
conda activate pipi-vision
```

已装：torch 2.14.0 / torchvision 0.29.0 / ultralytics 8.4.150 / clip / opencv / mediapipe 0.10.14

> **注意 Python 版本**：MediaPipe 官方只支持 3.9–3.12，你系统默认的 3.13 装不上。
> 这个环境是 3.11，正好。不要在 3.13 上重建。

首次运行会自动下载两个模型：

| 模型 | 大小 | 说明 |
|---|---|---|
| `yolov8s-world.pt` | 27 MB | 检测主干，下到项目根目录 |
| CLIP ViT-B/32 | 338 MB | 文本编码器，下到 `~/.cache/clip/` |

CLIP 下载较慢，耐心等一次，之后走缓存。

---

## 二、最快上手

### 方式 1：临时指定要找什么（调试 prompt 用）

```bash
cd pipi_vision

python detect.py --prompt "paper cup" "scissors" "hand" test_images/cup_pyramid.jpg
```

### 方式 2：用活动配置（正式用法）

```bash
python detect.py --config activities/configs/paper_cup.yaml test_images/*.jpg
```

### 常用参数

| 参数 | 作用 | 建议 |
|---|---|---|
| `--min-conf 0.01` | 统一压低阈值 | **标定时必用**，否则看不到低分候选，没法决定阈值 |
| `--imgsz 960` | 提高推理分辨率 | 找剪刀、笔、胶水这类小物体时必须调高 |
| `--size m` | 换更大的模型 | 精度不够时试 m/l，速度会掉 |
| `--device cpu` | 强制用 CPU | mps 出问题时回退 |
| `--out 目录` | 改输出位置 | 默认 `out/` |

### 输出

```
out/cup_pyramid_detected.jpg    # 带框、带中文名和置信度的标注图
out/cup_pyramid_detected.json   # 结构化结果，可直接喂给后续模块
```

JSON 结构：

```json
{
  "source": "test_images/cup_pyramid.jpg",
  "image_size": [3024, 4032],
  "device": "mps",
  "infer_ms": 180.5,
  "objects": [
    {
      "object_id": "paper_cup",
      "name_cn": "纸杯",
      "matched_prompt": "stacked paper cups",
      "confidence": 0.62,
      "bbox_xyxy": [1120, 2200, 1480, 2650],
      "center_xy": [1300, 2425],
      "area_ratio": 0.0489
    }
  ]
}
```

---

## 三、在代码里调用

```python
from vision.object_detector import ObjectDetector, ObjectSpec

detector = ObjectDetector(
    specs=[
        ObjectSpec(id="paper_cup", name_cn="纸杯",
                   prompts=["paper cup", "disposable cup"], conf=0.10),
        ObjectSpec(id="hand", name_cn="手", prompts=["hand"], conf=0.05),
    ],
    model_size="s",
    imgsz=640,
)

result = detector.detect("path/to/image.jpg")

cup = result.top("paper_cup")          # 置信度最高的纸杯
if cup:
    print(cup.bbox_xyxy, cup.center_xy, cup.confidence)

all_cups = result.by_object("paper_cup")   # 所有纸杯（堆叠场景要数个数）

print(result.to_dict())                # 完整结构化结果
```

**从活动配置加载：**

```python
import yaml
from vision.object_detector import ObjectDetector, specs_from_config

cfg = yaml.safe_load(open("activities/configs/paper_cup.yaml", encoding="utf-8"))
detector = ObjectDetector(specs_from_config(cfg["activity"]))
```

---

## 四、四个必须知道的坑

### 1. prompts 用英文，不要用中文

YOLO-World 的文本编码器是 CLIP，训练语料以英文为主。
`"paper cup"` 的效果显著好于 `"纸杯"`。

配置里 `name_cn` 只用于界面展示，真正喂给模型的是 `prompts`。

### 2. 一个物体配多个 prompt，取最高分

不同角度、不同材质下，同一个物体的最佳描述不一样：

```yaml
- id: paper_cup
  prompts:
    - paper cup          # 通用
    - disposable cup     # 强调一次性
    - white paper cup    # 强调白色
    - stacked paper cups # 强调堆叠状态
```

代码里会自动把同一 `id` 的多个 prompt 结果合并，只保留最高分那个。
**代价**：prompt 越多，类别数越多，推理越慢。当前配置 16 个 prompt，属于可接受范围。

### 3. `set_classes()` 绝对不能逐帧调用

这是最容易写出性能灾难的地方。该调用会重算 CLIP 文本 embedding，耗时数百毫秒。

- ✅ 活动开始时调一次（构造函数里已做）
- ✅ 切换到别的活动时调一次
- ❌ 每帧调用 → 帧率直接掉到 1 FPS 以下

代码里 `ObjectDetector` 已经把它锁在 `__init__`，正常用不会踩到。

### 4. 置信度阈值要比普通 YOLO 低很多

开放词汇检测的分数分布和固定类别检测完全不同：
普通 YOLOv8 上 0.5 算高分，YOLO-World 上 **0.15～0.3 往往就是正确检出**，
密集场景里的小目标甚至只有 0.02~0.06（实测见第五节）。

所以配置里各类阈值是分开定的：纸杯 0.02（要计数）、手 0.10、
桌子 0.15、剪刀/胶水/彩笔 0.08。正确做法：

```bash
# 先用极低阈值看真实分数分布
python detect.py --config ... --min-conf 0.01 test_images/*.jpg
```

看输出里每类物体的分数，再决定收紧到多少。
**不要凭感觉定阈值，要看自己素材上的真实分布。**

---

## 五、实测结果（2026-09-14，3 张实拍图）

设备：MacBook Pro M5 / 32GB，**CPU 推理**（mps 在受限执行环境里不可用；
你自己在终端跑 `--device auto` 会走 MPS，应比下面更快）。
模型：yolov8s-world，imgsz=640，dup_iou=0.65。

| 图片 | 纸杯 | 手 | 桌子 | 单张推理 |
|---|---|---|---|---|
| 近景单人搭 6 杯 | 6 个（真值 6）✅ | 1/2 只 ⚠ | 2 ✅ | 56 ms |
| 纸杯金字塔 | 32 个 ✅ | 0 ⚠ | 1 ✅ | 56 ms |
| 教室杯塔（多人） | 58 个 ✅ | 3 ✅ | 2 ✅ | 54 ms |

**结论 1：纸杯检测可用，且能计数。**
清晰近景 0.34~0.84，密集杯塔里的小杯 0.02~0.06。
计数必须用低阈值（配置里定 0.02），代价是混入少量误检，靠后续几何/跟踪过滤。

**结论 2：不要用 YOLO-World 找手。**
真手置信度只有 0.1~0.2，而橙色纸杯、孩子上半身都能触发 hand 类 prompt
（`child hand` 曾把孩子整个上半身框成一个占画面 44% 的"手"）。
手的正确来源是 **MediaPipe Hand**（方案第 6 节本来就这么设计），
配置里的 hand 段只是临时占位，接视频后删掉，改用手部关键点。

**结论 3：prompt 不是越多越好。**
实测删掉的三个 prompt：`child hand`（框上半身）、`crayon` 和
`colored pencil`（红色花纹纸杯被当成蜡笔/彩笔）。
保留多 prompt 的唯一理由：换个说法能捞回真漏检。

**结论 4：速度余量充足。**
CPU 上单张 55 ms 左右。接视频流时检测跑 2~5 FPS 就够，中间帧交给 tracking 补。

标定命令（换素材后重跑一遍，用真实分布决定阈值）：

```bash
python detect.py --config activities/configs/paper_cup.yaml \
  --min-conf 0.01 --dup-iou 0.65 你的图片*.jpg
```

---

## 六、接下来要测的四件事（实验 2 的验收项）

方案文档第 23 节列了实验 2 的观察点，这里给成可执行的测试：

| # | 测什么 | 怎么测 | 关注指标 |
|---|---|---|---|
| 1 | 正视角检出率 | 拍 10 张标准俯拍/45° 的纸杯图 | 检出几张？分数多少？ |
| 2 | 多角度稳定性 | 同一堆杯子，转 4 个角度各拍一张 | 分数波动大不大 |
| 3 | **遮挡表现** | 手握住杯子时拍 | 能不能保住？这是第一大失败模式 |
| 4 | 多目标区分 | 桌上放 5～8 个杯子 | `by_object("paper_cup")` 返回几个？漏了几个？ |

同时记录 **实测 FPS**：看 JSON 里的 `infer_ms`。
这个数决定了后面视频流的检测频率上限，必须实测，不能靠估。

> 提醒：`infer_ms` 是单张纯推理时间，不含读图和后处理。
> 接视频流时不需要每帧都检测——检测跑 2～5 FPS，中间帧交给 tracking 补。

---

## 七、目录结构

```
pipi_vision/
├── detect.py                      # 命令行调试脚本
├── yolov8s-world.pt               # 自动下载的检测权重
├── vision/
│   └── object_detector.py         # 检测器封装（本模块核心）
├── activities/
│   └── configs/
│       └── paper_cup.yaml         # 纸杯搭建的活动配置
├── test_images/                   # 测试图片
└── out/                           # 检测结果（标注图 + JSON）
```

按方案文档第 21 节的结构留好了位置，后续往里加：

```
vision/hand_detector.py      # MediaPipe Hand
vision/object_tracker.py     # ByteTrack，负责遮挡期间保住 track_id
engine/activity_engine.py    # 证据分数状态机
vlm/client.py                # 低频 VLM 调用
agent/agent.py               # 带节流的发言逻辑
```

---

## 八、追踪模块（track.py，2026-09-14 新增）

```bash
# 视频：检测+追踪，输出带 ID 标注视频 + 逐帧 JSONL（状态机的输入）
python track.py --video 杯子搭建.mp4 --config activities/configs/paper_cup.yaml \
  --min-conf 0.15 --detect-every 3 --min-hits 2

# 摄像头实时
python track.py --camera 0 --config ... --min-conf 0.15
```

**追踪阈值和检测阈值必须分开。** 检测标定用 0.02 是为了"看得见"分数
分布；追踪时用 0.02 会把"整杯框+半截框"的低分影子全放进来，轨迹数
爆炸（实测 15 条 vs 真实 3 个杯子）。0.15 起步，用你素材上
"真杯轨迹数 ≈ 桌上杯子数"来标定。

三层防御（都在 `vision/object_tracker.py`，各有单元测试守着）：

| 机制 | 治什么 | 关键参数 |
|---|---|---|
| confirmed 门控 | 单帧闪现误检 | `min_hits=3` |
| 恒速滑行 | 手完全挡住杯子时的断轨 | `max_coast=75`（≈2.5s@30fps） |
| 轨迹合并+新生抑制 | 同一杯子两条平行轨迹 | `merge_streak=10`（持续重叠才并，穿行交叉不误伤） |

合成视频实测（3 个真实纸杯贴片、1 个全程被挡 2 秒后平移 600px）：

- 稳态并存轨迹 3.08 条（真实 3），静止杯全片漂移 ≤2px
- 检测每 3 帧一次，MPS 单次检测 4ms，整段 92~112 FPS
- 已知边界：滑行后位置误差 ~100px 时，2 秒级完全遮挡的重新认领
  可能晚 1~2 帧（表现为短暂新 ID 后由合并逻辑收敛）。真实手持
  遮挡通常 <1s 且目标未完全消失，此场景更宽松。

回归命令：`python tests/test_tracker.py`（6 项）
