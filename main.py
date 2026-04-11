import cv2
import math
import threading
import time
from ultralytics import YOLO
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# ======================
# 🧠 模型加载
# ======================

human_model = pipeline(
    Tasks.domain_specific_object_detection,
    model='iic/cv_tinynas_human-detection_damoyolo',
    trust_remote_code=True
)

cigarette_model = pipeline(
    Tasks.domain_specific_object_detection,
    model='iic/cv_tinynas_object-detection_damoyolo_cigarette',
    trust_remote_code=True
)

mask_model = pipeline(
    Tasks.domain_specific_object_detection,
    model='iic/cv_tinynas_object-detection_damoyolo_facemask',
    trust_remote_code=True
)

fire_model = YOLO("fire.pt")  # 改成你的权重
pose_model = YOLO("yolov8s-pose.pt")

# ======================
# 🎥 视频
# ======================

cap = cv2.VideoCapture("fire.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = 1.0 / fps if fps > 0 else 0.033

latest_frame = None
lock = threading.Lock()

running = True
video_finished = False

# ======================
# 📦 结果缓存
# ======================

result_human = {}
result_cig = {}
result_mask = {}
result_fire = {}

# ======================
# 📊 状态追踪器
# ======================

sleep_tracker = {}   # person_id -> {"sleep_count": 0, "sleeping": False}
mask_tracker = {}    # person_id -> {"no_mask_count": 0, "no_mask": False}
person_counter = 0   # 用于给每个人分配 ID

# ======================
# 🎥 视频线程
# ======================

def video_stream():
    global latest_frame, running, video_finished

    while running:
        ret, frame = cap.read()

        if not ret:
            video_finished = True
            running = False
            break

        with lock:
            latest_frame = frame

        time.sleep(frame_interval)

# ======================
# 🧠 检测线程
# ======================

def detect_human():
    global result_human

    while running:
        if latest_frame is None:
            time.sleep(0.01)
            continue

        with lock:
            frame = latest_frame.copy()

        result_human = human_model(frame)


def detect_cigarette():
    global result_cig

    while running:
        if latest_frame is None:
            time.sleep(0.01)
            continue

        with lock:
            frame = latest_frame.copy()

        result_cig = cigarette_model(frame)


def detect_mask():
    global result_mask

    while running:
        if latest_frame is None:
            time.sleep(0.01)
            continue

        with lock:
            frame = latest_frame.copy()

        result_mask = mask_model(frame)


def detect_fire():
    global result_fire

    while running:
        if latest_frame is None:
            time.sleep(0.01)
            continue

        with lock:
            frame = latest_frame.copy()

        r = fire_model.predict(frame, conf=0.25, verbose=False)

        boxes = []
        scores = []

        if r and r[0].boxes is not None:
            for b in r[0].boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                boxes.append([x1, y1, x2, y2])
                scores.append(float(b.conf[0]))

        result_fire = {"boxes": boxes, "scores": scores}


def calc_iou(box1, box2):
    """计算两个框的 IoU，box = [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (box1_area + box2_area - inter_area)


def cleanup_trackers(tracker, alive_ids):
    """清理不再出现的人的追踪状态"""
    for pid in list(tracker.keys()):
        if pid not in alive_ids:
            del tracker[pid]


def detect_sleep():
    global person_counter, sleep_tracker

    while running:
        if latest_frame is None or "boxes" not in result_human:
            time.sleep(0.01)
            continue

        with lock:
            frame = latest_frame.copy()

        human_boxes = result_human["boxes"]
        human_scores = result_human["scores"]

        # 为每个检测到人分配唯一 ID
        alive_ids = set()
        person_boxes = []

        for box, score in zip(human_boxes, human_scores):
            pid = person_counter
            person_counter += 1
            alive_ids.add(pid)
            person_boxes.append((pid, box))

        # 清理已消失的人
        cleanup_trackers(sleep_tracker, alive_ids)
        cleanup_trackers(mask_tracker, alive_ids)

        for pid, box in person_boxes:
            x1, y1, x2, y2 = box
            # 裁剪人体区域给 pose 模型
            crop = frame[max(0, y1):min(frame.shape[0], y2),
                         max(0, x1):min(frame.shape[1], x2)]

            if crop.size == 0:
                continue

            r = pose_model.predict(crop, conf=0.25, verbose=False)

            is_sleeping = False
            if r and r[0].keypoints is not None:
                kpts = r[0].keypoints.xy[0]  # 17 个关键点
                if len(kpts) >= 7 and all(k > 0 for k in kpts[0]):
                    nose_y = kpts[0][1]
                    l_shoulder_y = kpts[5][1]
                    r_shoulder_y = kpts[6][1]
                    shoulder_y = (l_shoulder_y + r_shoulder_y) / 2

                    # 头部低于肩膀
                    if nose_y > shoulder_y:
                        is_sleeping = True

                    # 身体倾斜角（鼻子到髋部的角度）
                    l_hip_y = kpts[11][1]
                    r_hip_y = kpts[12][1]
                    hip_y = (l_hip_y + r_hip_y) / 2
                    nose_x = kpts[0][0]
                    hip_x = (kpts[11][0] + kpts[12][0]) / 2

                    if hip_y != nose_y:
                        angle = math.degrees(math.atan2(abs(hip_x - nose_x), abs(hip_y - nose_y)))
                        if angle > 60:
                            is_sleeping = True

            # 更新 sleep_tracker
            if pid not in sleep_tracker:
                sleep_tracker[pid] = {"sleep_count": 0, "sleeping": False}

            tracker = sleep_tracker[pid]
            if is_sleeping:
                tracker["sleep_count"] += 1
            else:
                tracker["sleep_count"] = max(0, tracker["sleep_count"] - 2)

            if tracker["sleep_count"] >= 150:
                tracker["sleeping"] = True

            if tracker["sleeping"] and tracker["sleep_count"] == 0:
                tracker["sleeping"] = False


# ======================
# 🚀 启动线程
# ======================

threading.Thread(target=video_stream, daemon=True).start()
threading.Thread(target=detect_human, daemon=True).start()
threading.Thread(target=detect_cigarette, daemon=True).start()
threading.Thread(target=detect_mask, daemon=True).start()
threading.Thread(target=detect_fire, daemon=True).start()
threading.Thread(target=detect_sleep, daemon=True).start()

# ======================
# 🎬 主循环
# ======================

while True:

    # ⭐ 统一退出条件（关键修复）
    if video_finished:
        running = False
        break

    if latest_frame is None:
        time.sleep(0.01)
        continue

    with lock:
        frame = latest_frame.copy()

    # 🧍 人
    if 'boxes' in result_human:
        for box, score in zip(result_human['boxes'], result_human['scores']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'person {score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 🚬 香烟
    if 'boxes' in result_cig:
        for box, score in zip(result_cig['boxes'], result_cig['scores']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'cig {score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 😷 口罩
    if 'boxes' in result_mask:
        for box, score in zip(result_mask['boxes'], result_mask['scores']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'mask {score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 🔥 火
    if 'boxes' in result_fire:
        for box, score in zip(result_fire['boxes'], result_fire['scores']):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, f'fire {score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.imshow("AI Surveillance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

# ======================
# 🧹 资源释放
# ======================

cap.release()
cv2.destroyAllWindows()