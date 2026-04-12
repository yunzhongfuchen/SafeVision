"""AI Video Surveillance Detection Engine"""

import cv2
import math
import base64
import threading
import time
from datetime import datetime
from ultralytics import YOLO
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


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
pose_model = YOLO("yolov8s-pose.pt")


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
mask_tracker = {}
person_counter = 0

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

def calc_iou(box1, box2):
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
    for pid in list(tracker.keys()):
        if pid not in alive_ids:
            del tracker[pid]


def match_persons(current_boxes, prev_boxes, iou_threshold=0.3):
    mapping = {}
    used_pids = set()
    for i, cbox in enumerate(current_boxes):
        best_pid = None
        best_iou = 0
        for pid, pbox in prev_boxes.items():
            if pid in used_pids:
                continue
            iou = calc_iou(cbox, pbox)
            if iou > best_iou:
                best_iou = iou
                best_pid = pid
        if best_pid is not None and best_iou > iou_threshold:
            used_pids.add(best_pid)
            mapping[i] = best_pid
    return mapping


def detect_sleep():
    global person_counter, sleep_tracker, result_sleep
    tick, get_fps = _track_thread_fps("detect_sleep")
    prev_boxes = {}
    while running:
        if latest_frame is None or "boxes" not in result_human:
            time.sleep(0.05)
            continue
        frame = latest_frame.copy()
        human_boxes = list(result_human["boxes"])
        human_scores = list(result_human["scores"])
        matched = match_persons(human_boxes, prev_boxes)
        alive_ids = set()
        person_boxes = []
        for i, (box, score) in enumerate(zip(human_boxes, human_scores)):
            if i in matched:
                pid = matched[i]
            else:
                pid = person_counter
                person_counter += 1
            alive_ids.add(pid)
            person_boxes.append((pid, box, score))
        prev_boxes.clear()
        for pid, box, _ in person_boxes:
            prev_boxes[pid] = list(box)
        cleanup_trackers(sleep_tracker, alive_ids)
        cleanup_trackers(mask_tracker, alive_ids)
        sleep_frames_threshold = get_config_value("sleep_frames") or 150
        conf_pose_val = get_config_value("conf_pose") or 0.25
        frame_results = []
        for pid, box, score in person_boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[max(0, y1):min(frame.shape[0], y2),
                         max(0, x1):min(frame.shape[1], x2)]
            if crop.size == 0:
                frame_results.append({"box": box, "sleeping": False, "score": score})
                continue
            try:
                r = pose_model.predict(crop, conf=conf_pose_val, verbose=False)
                is_sleeping = False
                if r and r[0].keypoints is not None and len(r[0].keypoints.xy) > 0:
                    kpts = r[0].keypoints.xy[0]
                    if len(kpts) >= 13 and all(k > 0 for k in kpts[0]):
                        nose_y = kpts[0][1]
                        l_shoulder_y = kpts[5][1]
                        r_shoulder_y = kpts[6][1]
                        shoulder_y = (l_shoulder_y + r_shoulder_y) / 2
                        if nose_y > shoulder_y:
                            is_sleeping = True
                        l_hip_y = kpts[11][1]
                        r_hip_y = kpts[12][1]
                        hip_y = (l_hip_y + r_hip_y) / 2
                        nose_x = kpts[0][0]
                        hip_x = (kpts[11][0] + kpts[12][0]) / 2
                        if hip_y != nose_y:
                            angle = math.degrees(math.atan2(abs(hip_x - nose_x), abs(hip_y - nose_y)))
                            if angle > 60:
                                is_sleeping = True
            except Exception:
                is_sleeping = False
            if pid not in sleep_tracker:
                sleep_tracker[pid] = {"sleep_count": 0, "sleeping": False}
            tracker = sleep_tracker[pid]
            if is_sleeping:
                tracker["sleep_count"] += 1
            else:
                tracker["sleep_count"] = max(0, tracker["sleep_count"] - 2)
            if tracker["sleep_count"] >= sleep_frames_threshold:
                tracker["sleeping"] = True
            if tracker["sleeping"] and tracker["sleep_count"] == 0:
                tracker["sleeping"] = False
            frame_results.append({"box": box, "sleeping": tracker["sleeping"], "score": score})
            if tracker["sleeping"]:
                snap = _capture_snapshot(frame, "sleep", score, box)
                add_log_entry("sleep", score, box, snap)
        result_sleep = frame_results
        tick()
        time.sleep(0.01)


# ======================
# Render Thread
# ======================


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
    global frame_interval, person_counter, video_fps, _start_time
    global sleep_tracker, mask_tracker, result_human, result_cig, result_mask
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
    person_counter = 0
    sleep_tracker.clear()
    mask_tracker.clear()
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
