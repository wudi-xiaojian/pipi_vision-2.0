import cv2
import time
import json
import torch
from vision.hand_detector import HandDetector
from vision.tracker import ObjectTracker
from perception_serializer import serialize_perception


def run_tracking_pipeline(video_source=0, output_jsonl="perception_output.jsonl"):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {video_source}")
        return

    # 初始化模块
    hand_detector = HandDetector()
    tracker = ObjectTracker(model_path="yolov11x-worldv2.pt")

    frame_index = 0
    start_time = time.perf_counter()

    # 按照 10Hz 频率序列化输出
    last_serialize_time = 0.0
    serialize_interval = 0.1  # 100ms

    jsonl_file = open(output_jsonl, "w", encoding="utf-8")
    print(f"Starting tracking... Saving logs to {output_jsonl}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or failed to read frame.")
                break

            frame_index += 1
            current_time = time.perf_counter()
            # 获取真实毫秒级时间戳（解决 MediaPipe 同步问题）
            timestamp_ms = int((current_time - start_time) * 1000)

            # 图像预处理（如需镜像处理，应先识别后再翻转显示，避免坐标污染；此处优先保持原始坐标一致性）

            # 禁用梯度构建，优化推理性能
            with torch.no_grad():
                # 1. 跑手部检测
                hands_data = hand_detector.detect(frame, timestamp_ms)

                # 2. 跑目标追踪 (YOLO-World + Kalman)
                tracked_objects = tracker.process(frame)

            # 3. 按 10Hz 进行数据序列化与导出
            if current_time - last_serialize_time >= serialize_interval:
                last_serialize_time = current_time
                perception_data = serialize_perception(
                    frame_index=frame_index,
                    timestamp=current_time - start_time,
                    hands_data=hands_data,
                    tracked_objects=tracked_objects
                )
                jsonl_file.write(json.dumps(perception_data, ensure_ascii=False) + "\n")
                jsonl_file.flush()

            # 4. 可视化绘制
            annotated_frame = frame.copy()
            hand_detector.draw(annotated_frame, hands_data)
            tracker.draw(annotated_frame, tracked_objects)

            # 最终展示视图支持镜像翻转，方便实时交互观看
            display_frame = cv2.flip(annotated_frame, 1)
            cv2.imshow("PiPi Tracking (q/Esc = stop)", display_frame)

            # 补全截断的按键检测逻辑
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("User requested stop.")
                break

    finally:
        # 补全资源释放逻辑
        cap.release()
        cv2.destroyAllWindows()
        jsonl_file.close()
        print(f"Pipeline closed cleanly. Processed {frame_index} frames.")


if __name__ == "__main__":
    run_tracking_pipeline(video_source=0)