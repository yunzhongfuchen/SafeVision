"""AI Video Surveillance Detection Engine"""

import cv2
import numpy as np
import base64
import threading
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import sleep_detect


# ======================
# Model Loading
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

fire_model = YOLO("fire.pt")
sleep_pose_model = YOLO("yolov8s-pose.pt")


# ======================
# Shared State
# ======================

# Video thread writes latest frame here
latest_frame = None

# Rendered annotated frame for UI
annotated_frame = None
annotated_jpeg = None  # pre-encoded JPEG bytes

running = False
video_finished = False

# Detection results
result_human = {}
result_cig = {}
result_mask = {}
result_no_mask = {}
result_fire = {}
result_smoke = {}
result_sleep = {}

# State trackers
sleep_tracker = {}

# Detection threads list (for cleanup on restart)
_detection_threads: list[threading.Thread] = []
_threads_lock = threading.Lock()

# Video source
cap = None
current_source = None
current_source_path = None
frame_interval = 0.033
video_fps = 0

# ======================
# System State & Thread FPS Tracking
# ======================

_start_time = 0.0

_thread_stats = {}
_thread_stats_lock = threading.Lock()


def _track_thread_fps(name):
    """Decorator-like helper: returns (tick, get_fps) for tracking."""
    frame_count = 0
    last_time = time.time()
    fps = 0.0

    def tick():
        nonlocal frame_count, last_time, fps
        frame_count += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps = frame_count / (now - last_time)
            frame_count = 0
            last_time = now

    def get_fps():
        return fps

    with _thread_stats_lock:
        _thread_stats[name] = {"running": True, "fps": fps, "get_fps": get_fps, "tick": tick}
    return tick, get_fps


def get_system_status():
    """Return system status dict for UI."""
    with _thread_stats_lock:
        thread_info = {}
        for name, info in _thread_stats.items():
            get_fps_fn = info.get("get_fps")
            thread_info[name] = {
                "running": running and info["running"],
                "fps": get_fps_fn() if get_fps_fn else 0.0,
            }

    uptime = time.time() - _start_time if _start_time > 0 else 0
    hrs = int(uptime // 3600)
    mins = int((uptime % 3600) // 60)
    secs = int(uptime % 60)

    source_label = "未连接"
    if current_source == "camera":
        source_label = "摄像头"
    elif current_source == "file":
        import os
        source_label = os.path.basename(current_source_path or "")

    return {
        "running": running,
        "source": source_label,
        "uptime": f"{hrs:02d}:{mins:02d}:{secs:02d}" if _start_time > 0 else "00:00:00",
        "video_fps": round(video_fps, 1),
        "threads": thread_info,
    }


# ======================
# Configuration
# ======================

_config = {
    "conf_fire": 0.25,
    "conf_pose": 0.25,
    "log_cooldown": 5.0,
    "sleep_frames": 150,
    "alerts": {
        "fire": True,
        "smoke": True,
        "cig": True,
        "no_mask": True,
        "sleep": True,
    },
}
_config_lock = threading.Lock()


def get_config():
    with _config_lock:
        return dict(_config)


def set_config(key, value):
    with _config_lock:
        if key in _config:
            _config[key] = value


def set_alert(event_type, enabled):
    with _config_lock:
        if event_type in _config["alerts"]:
            _config["alerts"][event_type] = enabled


def get_alerts():
    with _config_lock:
        return dict(_config["alerts"])


def get_config_value(key):
    with _config_lock:
        return _config.get(key)


# ======================
# Abnormal Event Logging
# ======================

class LogEntry:
    def __init__(self, event_type, score, box=None, snapshot=None):
        self.time = datetime.now().strftime("%H:%M:%S")
        self.event_type = event_type
        self.score = float(score)
        self.box = box
        self.snapshot = snapshot


log_entries: list[LogEntry] = []
log_lock = threading.Lock()
last_log_time = {}
LOG_COOLDOWN = 5.0


_SNAPSHOT_COLORS = {
    "fire": (0, 0, 255),
    "smoke": (0, 255, 255),
    "cig": (0, 0, 255),
    "no_mask": (0, 165, 255),
    "sleep": (255, 0, 255),
}

_SNAPSHOT_LABELS = {
    "fire": "Fire",
    "smoke": "Smoke",
    "cig": "Cigarette",
    "no_mask": "No Mask",
    "sleep": "Sleeping",
}


def _capture_snapshot(frame, event_type, score, box) -> str | None:
    """Capture current frame with detection box drawn, as base64 JPEG."""
    try:
        snap = frame.copy()
        color = _SNAPSHOT_COLORS.get(event_type, (0, 255, 0))
        label = _SNAPSHOT_LABELS.get(event_type, "Alert")
        if box is not None:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(snap, (x1, y1), (x2, y2), color, 3)
            cv2.putText(snap, f'{label} {score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        _, buf = cv2.imencode('.jpg', snap, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        return f'data:image/jpeg;base64,{b64}'
    except Exception:
        return None


def add_log_entry(event_type, score, box=None, snapshot=None):
    now = time.time()
    with _config_lock:
        cooldown = _config["log_cooldown"]
        alert_enabled = _config["alerts"].get(event_type, True)
    if not alert_enabled:
        return
    with log_lock:
        if event_type in last_log_time:
            if now - last_log_time[event_type] < cooldown:
                return
        last_log_time[event_type] = now
        log_entries.append(LogEntry(event_type, score, box, snapshot))


def get_log_entries(limit=50):
    with log_lock:
        return list(log_entries[-limit:])


def clear_logs():
    with log_lock:
        log_entries.clear()


def export_logs_csv():
    """Export logs as CSV string."""
    import io
    import csv
    with log_lock:
        entries = list(log_entries)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["时间", "类型", "置信度"])
    type_map = {
        "fire": "明火", "smoke": "烟雾", "cig": "抽烟",
        "no_mask": "未戴口罩", "sleep": "睡岗",
    }
    for entry in entries:
        writer.writerow([entry.time, type_map.get(entry.event_type, entry.event_type), f"{entry.score:.2f}"])
    return buf.getvalue()


# ======================
# Video Thread (reads frames at video rate, never blocked)
# ======================

def video_stream():
    global latest_frame, running, video_finished
    while running:
        if cap is None:
            time.sleep(0.01)
            continue
        ret, frame = cap.read()
        if not ret:
            video_finished = True
            running = False
            break
        latest_frame = frame
        time.sleep(frame_interval)


# ======================
# Detection Threads
# ======================

def detect_human():
    global result_human
    while running:
        if latest_frame is None:
            time.sleep(0.05)
            continue
        frame = latest_frame.copy()
        try:
            result_human = human_model(frame)
        except Exception:
            pass
        time.sleep(0.01)


def detect_cigarette():
    global result_cig
    while running:
        if latest_frame is None:
            time.sleep(0.05)
            continue
        frame = latest_frame.copy()
        try:
            r = cigarette_model(frame)
            result_cig = r
            if 'boxes' in r and len(r.get('boxes', [])) > 0:
                for box, score in zip(r['boxes'], r.get('scores', [])):
                    snap = _capture_snapshot(frame, "cig", score, box)
                    add_log_entry("cig", score, box, snap)
        except Exception:
            pass
        time.sleep(0.01)


def detect_mask():
    global result_mask, result_no_mask
    while running:
        if latest_frame is None:
            time.sleep(0.05)
            continue
        frame = latest_frame.copy()
        try:
            r = mask_model(frame)
            mask_boxes, mask_scores = [], []
            no_mask_boxes, no_mask_scores = [], []
            if 'boxes' in r and r['boxes'] is not None:
                for i, box in enumerate(r['boxes']):
                    score = r['scores'][i] if 'scores' in r else 0.5
                    label = str(r['labels'][i]) if 'labels' in r else 'facemask'
                    if label in ('facemask', '1'):
                        mask_boxes.append(list(box))
                        mask_scores.append(score)
                    elif label in ('no facemask', '2'):
                        no_mask_boxes.append(list(box))
                        no_mask_scores.append(score)
            result_mask = {"boxes": mask_boxes, "scores": mask_scores}
            result_no_mask = {"boxes": no_mask_boxes, "scores": no_mask_scores}
            if result_no_mask['boxes']:
                for box, score in zip(result_no_mask['boxes'], result_no_mask['scores']):
                    snap = _capture_snapshot(frame, "no_mask", score, box)
                    add_log_entry("no_mask", score, box, snap)
        except Exception:
            pass
        time.sleep(0.01)


def detect_fire():
    global result_fire, result_smoke
    tick, get_fps = _track_thread_fps("detect_fire")
    while running:
        if latest_frame is None:
            time.sleep(0.05)
            continue
        frame = latest_frame.copy()
        try:
            conf = get_config_value("conf_fire") or 0.25
            r = fire_model.predict(frame, conf=conf, verbose=False)
            fire_boxes, fire_scores = [], []
            smoke_boxes, smoke_scores = [], []
            if r and r[0].boxes is not None:
                for b in r[0].boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    cls = int(b.cls[0])
                    score = float(b.conf[0])
                    if cls == 0:
                        fire_boxes.append([x1, y1, x2, y2])
                        fire_scores.append(score)
                    else:
                        smoke_boxes.append([x1, y1, x2, y2])
                        smoke_scores.append(score)
            result_fire = {"boxes": fire_boxes, "scores": fire_scores}
            result_smoke = {"boxes": smoke_boxes, "scores": smoke_scores}
            for box, score in zip(fire_boxes, fire_scores):
                snap = _capture_snapshot(frame, "fire", score, box)
                add_log_entry("fire", score, box, snap)
            for box, score in zip(smoke_boxes, smoke_scores):
                snap = _capture_snapshot(frame, "smoke", score, box)
                add_log_entry("smoke", score, box, snap)
        except Exception:
            pass
        tick()
        time.sleep(0.01)


# ======================
# Helper Functions
# ======================

def detect_sleep():
    """睡岗检测线程 - 基于 YOLOv8-pose 全帧推理。"""
    global sleep_tracker, result_sleep
    tick, get_fps = _track_thread_fps("detect_sleep")
    sleep_frames_threshold = get_config_value("sleep_frames") or 150
    conf_pose_val = get_config_value("conf_pose") or 0.25
    sleep_tracker = {}
    prev_boxes = {}
    _pid_counter = 0

    while running:
        if latest_frame is None:
            time.sleep(0.05)
            continue
        frame = latest_frame.copy()

        # 全帧 pose 推理（YOLOv8-pose 自带人体检测）
        detections = sleep_detect.process_frame(sleep_pose_model, frame, conf=conf_pose_val)

        # 跨帧 ID 匹配（简单 IoU）
        current_boxes = [d['box'] for d in detections]
        mapping = {}
        used_pids = set()
        for i, cbox in enumerate(current_boxes):
            best_pid, best_iou = None, 0
            for pid, pbox in prev_boxes.items():
                if pid in used_pids:
                    continue
                x1m = max(cbox[0], pbox[0])
                y1m = max(cbox[1], pbox[1])
                x2m = min(cbox[2], pbox[2])
                y2m = min(cbox[3], pbox[3])
                inter = max(0, x2m - x1m) * max(0, y2m - y1m)
                if inter == 0:
                    continue
                area1 = (cbox[2] - cbox[0]) * (cbox[3] - cbox[1])
                area2 = (pbox[2] - pbox[0]) * (pbox[3] - pbox[1])
                iou = inter / (area1 + area2 - inter)
                if iou > best_iou:
                    best_iou = iou
                    best_pid = pid
            if best_pid is not None and best_iou > 0.3:
                used_pids.add(best_pid)
                mapping[i] = best_pid

        alive_ids = set()
        for i, det in enumerate(detections):
            pid = mapping.get(i)
            if pid is None:
                pid = _pid_counter
                _pid_counter += 1
            alive_ids.add(pid)
            prev_boxes[pid] = list(det['box'])

        for pid in list(sleep_tracker.keys()):
            if pid not in alive_ids:
                del sleep_tracker[pid]
        prev_boxes = {pid: box for pid, box in prev_boxes.items() if pid in alive_ids}

        frame_results = []
        for i, det in enumerate(detections):
            pid = mapping.get(i)
            if pid is None:
                pid = _pid_counter - 1
            if pid not in sleep_tracker:
                sleep_tracker[pid] = {"sleep_count": 0, "sleeping": False}
            tracker = sleep_tracker[pid]
            if det['sleeping']:
                tracker["sleep_count"] += 1
            else:
                tracker["sleep_count"] = max(0, tracker["sleep_count"] - 2)
            if tracker["sleep_count"] >= sleep_frames_threshold:
                tracker["sleeping"] = True
            elif tracker["sleep_count"] == 0:
                tracker["sleeping"] = False

            frame_results.append({
                "box": det['box'],
                "sleeping": tracker["sleeping"],
                "score": det['score'],
                "posture_label": det['posture_label'],
                "sleep_confidence": det['sleep_confidence'],
                "keypoints": det.get('keypoints'),
                "_info": det.get('_info', {}),
            })
            if tracker["sleeping"]:
                snap = _capture_snapshot(frame, "sleep", det['score'], det['box'])
                add_log_entry("sleep", det['score'], det['box'], snap)

        result_sleep = frame_results
        tick()
        time.sleep(0.01)


# ======================
# Render Thread
# ======================

FONT_PATH = 'C:/Windows/Fonts/msyh.ttc'
KPT_CONF_THRESHOLD = 0.4
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
KPT_COLORS = [
    (0, 0, 255), (255, 0, 0), (255, 0, 0), (255, 128, 0), (255, 128, 0),
    (0, 255, 0), (0, 255, 0), (0, 255, 128), (0, 255, 128),
    (128, 255, 0), (128, 255, 0), (255, 255, 0), (255, 255, 0),
    (0, 128, 255), (0, 128, 255), (255, 0, 255), (255, 0, 255),
]
POSTURE_LABELS = {
    'face_up': '仰卧',
    'face_down': '俯卧',
    'side': '侧卧',
    'standing/sitting': '站立/坐姿',
    'not sleeping': '未睡眠',
}


def load_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except:
        return ImageFont.load_default()


def draw_chinese_text(img, text, x, y, color=(255, 255, 255), font_size=22,
                      bg_color=None, padding=5):
    """使用 PIL 绘制中文文字，返回文字占用的宽高"""
    try:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, 'RGBA')
        font = load_font(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if bg_color:
            bg_rgba = (*bg_color, 200) if len(bg_color) == 3 else bg_color
            draw.rectangle(
                [x - padding, y - th - padding, x + tw + padding, y + padding],
                fill=bg_rgba
            )
        draw.text((x, y), text, fill=tuple(reversed(color)), font=font)
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return tw + padding * 2, th + padding * 2
    except Exception:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        return 0, 0


def _render_loop():
    """Read latest frame + detection results, draw boxes, cache JPEG for UI."""
    global annotated_frame, annotated_jpeg
    tick, get_fps = _track_thread_fps("_render_loop")
    while running:
        if latest_frame is None:
            time.sleep(0.01)
            continue
        frame = latest_frame.copy()
        r_cig = dict(result_cig)
        r_mask = dict(result_mask)
        r_nmask = dict(result_no_mask)
        r_fire = dict(result_fire)
        r_smoke = dict(result_smoke)

        for box, score in zip(r_cig.get('boxes', []), r_cig.get('scores', [])):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'cig {score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for box, score in zip(r_mask.get('boxes', []), r_mask.get('scores', [])):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'mask {score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for box, score in zip(r_nmask.get('boxes', []), r_nmask.get('scores', [])):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, f'no mask {score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        for box, score in zip(r_fire.get('boxes', []), r_fire.get('scores', [])):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'fire {score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for box, score in zip(r_smoke.get('boxes', []), r_smoke.get('scores', [])):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f'smoke {score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Sleep detection boxes + skeleton
        for entry in result_sleep:
            box = entry['box']
            x1, y1, x2, y2 = [int(v) for v in box]
            info = entry.get('_info', {})
            kp = entry.get('keypoints')
            posture_cn = POSTURE_LABELS.get(info.get('posture', ''), '')

            if info.get('is_sleeping'):
                box_color = (0, 255, 255)
                status_text = "[ 睡眠中 ]  " + posture_cn
                status_color = (0, 255, 255)
            else:
                box_color = (0, 0, 255)
                status_text = "[ 未睡眠 ]  " + posture_cn
                status_color = (100, 100, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

            # 状态行
            _, h1 = draw_chinese_text(frame, status_text, x1, max(0, y1 - 60),
                                       color=status_color, font_size=22,
                                       bg_color=(0, 0, 0))

            # 置信度行
            conf_text = f"睡眠: {info.get('sleep_confidence', 0):.0%}  |  睡姿: {info.get('posture_confidence', 0):.0%}"
            draw_chinese_text(frame, conf_text, x1, max(0, y1 - 60) + h1 + 2,
                              color=(220, 220, 220), font_size=18,
                              bg_color=(0, 0, 0))

            # 骨架
            if kp is not None and len(kp) >= 17:
                for a, b in SKELETON:
                    if float(kp[a, 2]) > KPT_CONF_THRESHOLD and float(kp[b, 2]) > KPT_CONF_THRESHOLD:
                        pt1 = (int(kp[a, 0]), int(kp[a, 1]))
                        pt2 = (int(kp[b, 0]), int(kp[b, 1]))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 3)
                        cv2.line(frame, pt1, pt2, (0, 180, 0), 1)
                for i in range(len(kp)):
                    if float(kp[i, 2]) > KPT_CONF_THRESHOLD:
                        x, y = int(kp[i, 0]), int(kp[i, 1])
                        cv2.circle(frame, (x, y), 6, KPT_COLORS[i], -1)
                        cv2.circle(frame, (x, y), 8, (255, 255, 255), 1)

        annotated_frame = frame
        try:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            annotated_jpeg = jpeg.tobytes()
        except Exception:
            annotated_jpeg = None

        tick()
        time.sleep(frame_interval)


# ======================
# Control Functions
# ======================

def start_detection(source_type, source_path=None):
    global cap, running, video_finished, current_source, current_source_path
    global frame_interval, video_fps, _start_time
    global sleep_tracker, result_human, result_cig, result_mask
    global result_no_mask, result_fire, result_smoke, result_sleep, annotated_frame, annotated_jpeg
    global latest_frame

    # Signal stop and wait for existing threads
    running = False
    video_finished = False

    with _threads_lock:
        for t in _detection_threads:
            t.join(timeout=2.0)
        _detection_threads.clear()

    if cap is not None:
        cap.release()

    time.sleep(0.1)

    if source_type == "file" and source_path:
        cap = cv2.VideoCapture(source_path)
    elif source_type == "camera":
        cap = cv2.VideoCapture(0)
    else:
        return "Please select a file or camera"

    if not cap.isOpened():
        return "Unable to open video source"

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1.0 / fps if fps > 0 else 0.033
    video_fps = fps

    latest_frame = None
    annotated_frame = None
    annotated_jpeg = None
    sleep_tracker.clear()
    result_human.clear()
    result_cig.clear()
    result_mask.clear()
    result_no_mask.clear()
    result_fire.clear()
    result_smoke.clear()
    result_sleep.clear()

    current_source = source_type
    current_source_path = source_path
    _start_time = time.time()
    running = True

    # Reset thread stats
    with _thread_stats_lock:
        _thread_stats.clear()

    thread_fns = [
        ("video_stream", video_stream),
        ("detect_human", detect_human),
        ("detect_cig", detect_cigarette),
        ("detect_mask", detect_mask),
        ("detect_fire", detect_fire),
        ("detect_sleep", detect_sleep),
        ("_render_loop", _render_loop),
    ]

    with _threads_lock:
        for name, fn in thread_fns:
            t = threading.Thread(target=fn, daemon=True, name=name)
            t.start()
            _detection_threads.append(t)

    return "Detection started"


def stop_detection():
    global running, video_finished
    running = False
    video_finished = False
    return "Detection stopped"


def get_stats():
    return {
        "fire": len(result_fire.get('boxes', [])),
        "smoke": len(result_smoke.get('boxes', [])),
        "cig": len(result_cig.get('boxes', [])),
        "no_mask": len(result_no_mask.get('boxes', [])),
        "sleep": sum(1 for s in result_sleep if s.get("sleeping")),
    }
