import cv2
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

# ======================
# 🚀 启动线程
# ======================

threading.Thread(target=video_stream, daemon=True).start()
threading.Thread(target=detect_human, daemon=True).start()
threading.Thread(target=detect_cigarette, daemon=True).start()
threading.Thread(target=detect_mask, daemon=True).start()
threading.Thread(target=detect_fire, daemon=True).start()

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