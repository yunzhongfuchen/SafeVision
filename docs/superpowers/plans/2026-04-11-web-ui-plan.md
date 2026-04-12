# AI 视频监控 Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 Gradio 构建双页 Web 监控界面，模型预加载，支持文件/摄像头实时检测

**Architecture:** 提取检测引擎到 engine.py，app.py 负责 Gradio UI 渲染。检测线程持续写入共享状态，Gradio 定时读取并更新 UI。

**Tech Stack:** Gradio 5.x, OpenCV, Ultralytics YOLO, ModelScope DAMOYOLO, Python threading

---

### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `engine.py` | 创建 | 检测引擎：模型加载、检测线程、共享状态、日志 |
| `app.py` | 创建 | Gradio Web UI：双页界面、事件绑定、视频流 |
| `main.py` | 修改 | 导入 engine.py（保持终端入口兼容性） |

---

### Task 1: 提取检测引擎到 engine.py

**Files:**
- Create: `engine.py`

**Responsibility:** 包含所有模型加载、检测线程函数、共享状态管理、异常日志记录。

- [ ] **Step 1: 创建 engine.py**

```python
"""AI 视频监控检测引擎 — 模型加载 + 检测线程 + 共享状态"""

import cv2
import math
import threading
import time
from datetime import datetime
from ultralytics import YOLO
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


# ======================
# 🧠 模型加载（启动时一次性加载）
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
# 📦 共享状态（检测线程写入，UI 读取）
# ======================

lock = threading.Lock()
latest_frame = None  # 当前视频帧（带标注）
running = False      # 是否正在检测
video_finished = False  # 视频是否播放完毕

# 检测结果缓存
result_human = {}
result_cig = {}
result_mask = {}
result_no_mask = {}
result_fire = {}
result_smoke = {}
result_sleep = {}  # [{box, sleeping, score}]

# 状态追踪器
sleep_tracker = {}
mask_tracker = {}
person_counter = 0

# 视频源
cap = None
current_source = None  # "file" or "camera"
frame_interval = 0.033


# ======================
# 📋 异常日志
# ======================

class LogEntry:
    def __init__(self, event_type, score, box=None, snapshot=None):
        self.time = datetime.now().strftime("%H:%M:%S")
        self.event_type = event_type  # "fire", "smoke", "cig", "no_mask", "sleep"
        self.score = score
        self.box = box  # [x1, y1, x2, y2]
        self.snapshot = snapshot  # 裁剪的异常区域

# 内存日志列表
log_entries: list[LogEntry] = []
log_lock = threading.Lock()
# 去重：记录上次写入时间 {event_type: timestamp}
last_log_time = {}
LOG_COOLDOWN = 5.0  # 同类异常 5 秒内只记录一次


def add_log_entry(event_type, score, box=None, snapshot=None):
    """添加异常日志（带去重）"""
    now = time.time()
    if event_type in last_log_time:
        if now - last_log_time[event_type] < LOG_COOLDOWN:
            return
    last_log_time[event_type] = now

    entry = LogEntry(event_type, score, box, snapshot)
    with log_lock:
        log_entries.append(entry)


def get_log_entries(limit=50):
    """获取最近的日志"""
    with log_lock:
        return list(log_entries[-limit:])


def clear_logs():
    """清空日志"""
    with log_lock:
        log_entries.clear()


# ======================
# 🎥 视频线程
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
        # 记录抽烟日志
        if 'boxes' in result_cig and result_cig['boxes']:
            for box, score in zip(result_cig['boxes'], result_cig['scores']):
                add_log_entry("cig", score, box)


def detect_mask():
    global result_mask, result_no_mask
    while running:
        if latest_frame is None:
            time.sleep(0.01)
            continue
        with lock:
            frame = latest_frame.copy()
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
        # 记录未戴口罩日志
        if result_no_mask['boxes']:
            for box, score in zip(result_no_mask['boxes'], result_no_mask['scores']):
                add_log_entry("no_mask", score, box)


def detect_fire():
    global result_fire, result_smoke
    while running:
        if latest_frame is None:
            time.sleep(0.01)
            continue
        with lock:
            frame = latest_frame.copy()
        r = fire_model.predict(frame, conf=0.25, verbose=False)
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
        # 记录火和烟日志
        for box, score in zip(fire_boxes, fire_scores):
            add_log_entry("fire", score, box)
        for box, score in zip(smoke_boxes, smoke_scores):
            add_log_entry("smoke", score, box)


# ======================
# 🛠 辅助函数
# ======================

def calc_iou(box1, box2):
    """计算两个框的 IoU"""
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


def match_persons(current_boxes, prev_boxes, iou_threshold=0.3):
    """基于 IoU 匹配当前帧的人体框到上一帧的 ID"""
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
    prev_boxes = {}
    while running:
        if latest_frame is None or "boxes" not in result_human:
            time.sleep(0.01)
            continue
        with lock:
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
        frame_results = []
        for pid, box, score in person_boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[max(0, y1):min(frame.shape[0], y2),
                         max(0, x1):min(frame.shape[1], x2)]
            if crop.size == 0:
                frame_results.append({"box": box, "sleeping": False, "score": score})
                continue
            r = pose_model.predict(crop, conf=0.25, verbose=False)
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
            frame_results.append({"box": box, "sleeping": tracker["sleeping"], "score": score})
            # 记录睡岗日志
            if tracker["sleeping"]:
                add_log_entry("sleep", score, box)
        with lock:
            result_sleep = frame_results
        time.sleep(0.01)


# ======================
# 🚀 渲染帧（带标注，供 UI 读取）
# ======================

def render_frame():
    """在当前帧上绘制所有检测标注，返回标注后的帧"""
    with lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        return None

    # 🧍 人体
    if 'boxes' in result_human:
        with lock:
            sleep_results = list(result_sleep)
        for i, (box, score) in enumerate(zip(result_human['boxes'], result_human['scores'])):
            x1, y1, x2, y2 = map(int, box)
            is_sleeping = False
            if i < len(sleep_results):
                is_sleeping = sleep_results[i]["sleeping"]
            if is_sleeping:
                label = f'SLEEP {score:.2f}'
                color = (0, 255, 255)
            else:
                label = f'person {score:.2f}'
                color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 🚬 香烟
    if 'boxes' in result_cig:
        for box, score in zip(result_cig['boxes'], result_cig['scores']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'cig {score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 😷 戴口罩
    if 'boxes' in result_mask:
        for box, score in zip(result_mask['boxes'], result_mask['scores']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'mask {score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 🚫 未戴口罩
    if 'boxes' in result_no_mask:
        for box, score in zip(result_no_mask['boxes'], result_no_mask['scores']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, f'no mask {score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # 🔥 火
    if 'boxes' in result_fire:
        for box, score in zip(result_fire['boxes'], result_fire['scores']):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'fire {score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 🌫️ 烟
    if 'boxes' in result_smoke:
        for box, score in zip(result_smoke['boxes'], result_smoke['scores']):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f'smoke {score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame


# ======================
# 🎬 控制函数（供 UI 调用）
# ======================

def start_detection(source_type, source_path=None):
    """启动检测。source_type: "file" 或 "camera"，source_path: 文件路径或摄像头索引"""
    global cap, running, video_finished, current_source, frame_interval, person_counter
    global sleep_tracker, mask_tracker, result_human, result_cig, result_mask
    global result_no_mask, result_fire, result_smoke, result_sleep, latest_frame

    # 重置状态
    if cap is not None:
        cap.release()
    running = False
    video_finished = False
    time.sleep(0.1)

    # 清理之前的线程
    for t in threading.enumerate():
        if t.name != "MainThread" and t.daemon:
            pass  # 旧线程会因 running=False 自动退出

    if source_type == "file" and source_path:
        cap = cv2.VideoCapture(source_path)
    elif source_type == "camera":
        cap = cv2.VideoCapture(0)
    else:
        return "请选择文件或摄像头"

    if not cap.isOpened():
        return "无法打开视频源"

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1.0 / fps if fps > 0 else 0.033

    # 重置检测状态
    with lock:
        latest_frame = None
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
    running = True

    # 启动线程
    threading.Thread(target=video_stream, daemon=True, name="video_stream").start()
    threading.Thread(target=detect_human, daemon=True, name="detect_human").start()
    threading.Thread(target=detect_cigarette, daemon=True, name="detect_cig").start()
    threading.Thread(target=detect_mask, daemon=True, name="detect_mask").start()
    threading.Thread(target=detect_fire, daemon=True, name="detect_fire").start()
    threading.Thread(target=detect_sleep, daemon=True, name="detect_sleep").start()

    return "检测已启动"


def stop_detection():
    """停止检测"""
    global running, video_finished
    running = False
    video_finished = False
    return "检测已停止"


def get_stats():
    """获取当前统计数据"""
    fire_count = len(result_fire.get('boxes', []))
    smoke_count = len(result_smoke.get('boxes', []))
    cig_count = len(result_cig.get('boxes', []))
    no_mask_count = len(result_no_mask.get('boxes', []))
    sleep_count = sum(1 for s in result_sleep if s.get("sleeping"))
    return {
        "fire": fire_count,
        "smoke": smoke_count,
        "cig": cig_count,
        "no_mask": no_mask_count,
        "sleep": sleep_count,
    }
```

- [ ] **Step 2: 验证 engine.py 可导入**

```bash
python -c "import engine; print('engine loaded')"
```

Expected: Models load, prints "engine loaded"

---

### Task 2: 创建 Gradio 监控页面

**Files:**
- Create: `app.py`

**Responsibility:** Gradio Web UI，包含实时监控页和后台管理页。

- [ ] **Step 1: 创建 app.py（监控页部分）**

```python
"""AI 视频监控 Web UI — Gradio 双页界面"""

import cv2
import numpy as np
import time
import gradio as gr
import engine


# ======================
# 🎨 浅色主题 CSS
# ======================

CUSTOM_CSS = """
/* 统计卡片 */
.stat-card {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 8px;
    margin: 4px;
    font-size: 14px;
    font-weight: 600;
}
.stat-fire { background: #fee2e2; color: #dc2626; }
.stat-smoke { background: #fef3c7; color: #d97706; }
.stat-cig { background: #d1fae5; color: #059669; }
.stat-mask { background: #dbeafe; color: #2563eb; }
.stat-sleep { background: #ede9fe; color: #7c3aed; }

/* 日志条目 */
.log-entry { padding: 4px 8px; margin: 2px 0; border-radius: 4px; font-size: 12px; }
.log-fire { background: #fee2e2; color: #dc2626; border-left: 3px solid #dc2626; }
.log-smoke { background: #fef3c7; color: #d97706; border-left: 3px solid #d97706; }
.log-cig { background: #fef3c7; color: #d97706; border-left: 3px solid #d97706; }
.log-no_mask { background: #dbeafe; color: #2563eb; border-left: 3px solid #2563eb; }
.log-sleep { background: #ede9fe; color: #7c3aed; border-left: 3px solid #7c3aed; }

/* 页面整体 */
.monitor-container { max-width: 1200px; margin: 0 auto; }
.gradio-container { background: #f5f7fa; }
"""

# ======================
# 📺 页面一：实时监控
# ======================

STATS_HTML = """
<div style="text-align:center; margin-bottom:10px;">
    <span class="stat-card stat-fire">🔥 火: <span id="fire-count">0</span></span>
    <span class="stat-card stat-smoke">🌫️ 烟: <span id="smoke-count">0</span></span>
    <span class="stat-card stat-cig">🚬 抽烟: <span id="cig-count">0</span></span>
    <span class="stat-card stat-mask">😷 未戴: <span id="mask-count">0</span></span>
    <span class="stat-card stat-sleep">💤 睡岗: <span id="sleep-count">0</span></span>
</div>
"""

LOG_HTML = """<div style="padding:10px; color:#6b7280;">等待检测启动...</div>"""

with gr.Blocks(title="AI 视频监控", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 AI 视频监控系统")

    with gr.Tabs():
        with gr.TabItem("📺 实时监控"):
            with gr.Row():
                # 左侧：视频 + 控制
                with gr.Column(scale=3):
                    stats_display = gr.HTML(STATS_HTML)
                    video_output = gr.Image(label="实时监控", type="numpy", interactive=False)
                    with gr.Row():
                        file_input = gr.File(label="选择视频文件", file_types=[".mp4", ".avi", ".mkv"])
                        play_btn = gr.Button("▶ 播放", variant="primary")
                        pause_btn = gr.Button("⏸ 暂停", variant="secondary")
                        camera_btn = gr.Button("📷 摄像头", variant="secondary")
                    status_display = gr.Textbox(label="状态", interactive=False)

                # 右侧：日志
                with gr.Column(scale=1):
                    log_display = gr.HTML(LOG_HTML)
                    log_link = gr.Markdown("[查看完整日志 → 后台管理](后台管理)", elem_id="log-link")

        with gr.TabItem("⚙️ 后台管理"):
            pass  # Task 3 实现


# ======================
# 🔌 事件处理
# ======================

def update_video():
    """Gradio 流函数：持续返回标注帧"""
    while True:
        if engine.running:
            frame = engine.render_frame()
            if frame is not None:
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield None
        time.sleep(0.05)


def update_stats():
    """更新统计卡片"""
    stats = engine.get_stats()
    return f"""
    <div style="text-align:center; margin-bottom:10px;">
        <span class="stat-card stat-fire">🔥 火: {stats['fire']}</span>
        <span class="stat-card stat-smoke">🌫️ 烟: {stats['smoke']}</span>
        <span class="stat-card stat-cig">🚬 抽烟: {stats['cig']}</span>
        <span class="stat-card stat-mask">😷 未戴: {stats['no_mask']}</span>
        <span class="stat-card stat-sleep">💤 睡岗: {stats['sleep']}</span>
    </div>
    """


def update_logs():
    """更新日志流"""
    entries = engine.get_log_entries(limit=10)
    if not entries:
        return '<div style="padding:10px; color:#6b7280;">暂无异常记录</div>'

    type_map = {
        "fire": ("🔥 明火", "log-fire"),
        "smoke": ("🌫️ 烟雾", "log-smoke"),
        "cig": ("🚬 抽烟", "log-cig"),
        "no_mask": ("😷 未戴口罩", "log-no_mask"),
        "sleep": ("💤 睡岗", "log-sleep"),
    }

    html = '<div style="padding:8px;">'
    for entry in reversed(entries):
        label, cls = type_map.get(entry.event_type, ("未知", ""))
        html += f'<div class="log-entry {cls}">{entry.time} {label} ({entry.score:.2f})</div>'
    html += '</div>'
    return html


def play_file(file_obj):
    """播放文件"""
    if file_obj is None:
        return "请先选择视频文件"
    result = engine.start_detection("file", file_obj.name)
    return result


def play_camera():
    """启动摄像头"""
    return engine.start_detection("camera")


def pause():
    """暂停"""
    return engine.stop_detection()


# 绑定事件
play_btn.click(play_file, inputs=[file_input], outputs=[status_display])
camera_btn.click(play_camera, outputs=[status_display])
pause_btn.click(pause, outputs=[status_display])

# 视频流：持续更新
demo.load(update_video, outputs=[video_output])

# 定时更新统计和日志（每 2 秒）
timer = gr.Timer(value=2)
timer.tick(update_stats, outputs=[stats_display])
timer.tick(update_logs, outputs=[log_display])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

- [ ] **Step 2: 验证 app.py 可启动**

```bash
python app.py
```

Expected: Gradio starts at http://localhost:7860

---

### Task 3: 完善后台管理页

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 替换后台管理页内容**

将 `with gr.TabItem("⚙️ 后台管理"):` 下的 `pass` 替换为：

```python
        with gr.TabItem("⚙️ 后台管理"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 统计汇总")
                    admin_stats = gr.HTML(
                        """<div style="text-align:center; color:#6b7280;">等待数据...</div>"""
                    )

                with gr.Column(scale=3):
                    gr.Markdown("### 📋 完整日志")
                    with gr.Row():
                        log_filter = gr.Dropdown(
                            choices=["全部", "🔥 明火", "🌫️ 烟雾", "🚬 抽烟", "😷 未戴口罩", "💤 睡岗"],
                            value="全部",
                            label="筛选类型",
                        )
                        clear_log_btn = gr.Button("🗑️ 清空日志")
                    admin_log = gr.HTML(
                        """<div style="padding:10px; color:#6b7280;">暂无日志</div>"""
                    )

            gr.Markdown("---")
            gr.Markdown("### ⚙️ 模型配置")
            with gr.Row():
                conf_fire = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="火焰/烟雾检测阈值")
                conf_pose = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="睡岗检测阈值")
```

- [ ] **Step 2: 添加后台管理页事件处理**

在 `if __name__ == "__main__":` 之前添加：

```python
def update_admin_stats():
    """更新后台统计汇总"""
    logs = engine.get_log_entries()
    counts = {"fire": 0, "smoke": 0, "cig": 0, "no_mask": 0, "sleep": 0}
    for entry in logs:
        counts[entry.event_type] = counts.get(entry.event_type, 0) + 1

    return f"""
    <div style="display:flex; gap:8px; justify-content:center; padding:10px;">
        <div style="background:#fee2e2; padding:10px 16px; border-radius:8px; text-align:center; flex:1;">
            <div style="font-size:20px; color:#dc2626; font-weight:700;">{counts['fire']}</div>
            <div style="font-size:10px; color:#991b1b;">火警</div>
        </div>
        <div style="background:#fef3c7; padding:10px 16px; border-radius:8px; text-align:center; flex:1;">
            <div style="font-size:20px; color:#d97706; font-weight:700;">{counts['smoke']}</div>
            <div style="font-size:10px; color:#92400e;">烟雾</div>
        </div>
        <div style="background:#d1fae5; padding:10px 16px; border-radius:8px; text-align:center; flex:1;">
            <div style="font-size:20px; color:#059669; font-weight:700;">{counts['cig']}</div>
            <div style="font-size:10px; color:#065f46;">抽烟</div>
        </div>
        <div style="background:#ede9fe; padding:10px 16px; border-radius:8px; text-align:center; flex:1;">
            <div style="font-size:20px; color:#7c3aed; font-weight:700;">{counts['sleep']}</div>
            <div style="font-size:10px; color:#5b21b6;">睡岗</div>
        </div>
    </div>
    """


def update_admin_logs(filter_type="全部"):
    """更新后台完整日志"""
    entries = engine.get_log_entries()
    if not entries:
        return '<div style="padding:10px; color:#6b7280;">暂无日志</div>'

    type_map = {
        "fire": ("🔥 明火", "log-fire"),
        "smoke": ("🌫️ 烟雾", "log-smoke"),
        "cig": ("🚬 抽烟", "log-cig"),
        "no_mask": ("😷 未戴口罩", "log-no_mask"),
        "sleep": ("💤 睡岗", "log-sleep"),
    }

    type_filter = None
    for k, (label, _) in type_map.items():
        if label == filter_type:
            type_filter = k
            break

    html = '<div style="padding:4px;">'
    html += '<div style="display:flex; justify-content:space-between; color:#6b7280; font-size:11px; padding:0 8px; margin-bottom:4px;">'
    html += '<span>时间</span><span>类型</span><span>置信度</span>'
    html += '</div>'

    for entry in reversed(entries):
        if type_filter and entry.event_type != type_filter:
            continue
        label, cls = type_map.get(entry.event_type, ("未知", ""))
        html += f'<div class="log-entry {cls}" style="display:flex; justify-content:space-between;">'
        html += f'<span>{entry.time}</span><span>{label}</span><span>{entry.score:.2f}</span>'
        html += '</div>'
    html += '</div>'
    return html


def do_clear_logs():
    engine.clear_logs()
    return '<div style="padding:10px; color:#6b7280;">日志已清空</div>'


# 后台页事件
log_filter.change(update_admin_logs, inputs=[log_filter], outputs=[admin_log])
clear_log_btn.click(do_clear_logs, outputs=[admin_log])
timer.tick(update_admin_stats, outputs=[admin_stats])
```

---

### Task 4: 修改 main.py 导入 engine

**Files:**
- Modify: `main.py`

- [ ] **Step 1: 简化 main.py，导入 engine**

将 `main.py` 替换为：

```python
"""AI 视频监控 — 终端入口（保留兼容）"""
import cv2
import threading
import time
import engine

# 使用默认视频文件
engine.start_detection("file", "smoking.mp4")

while engine.running:
    frame = engine.render_frame()
    if frame is not None:
        cv2.imshow("AI Surveillance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        engine.stop_detection()
        break

engine.cap.release()
cv2.destroyAllWindows()
```

---

### Task 5: 端到端验证

- [ ] **Step 1: 启动 Web UI**

```bash
python app.py
```

Expected output:
```
Running on local URL:  http://0.0.0.0:7860
```

- [ ] **Step 2: 访问页面**

打开浏览器 http://localhost:7860

- [ ] **Step 3: 验证监控页**
- 顶部显示 5 个统计卡片
- 可以选择文件并播放
- 视频画面实时显示带标注的帧
- 右侧显示异常日志流
- 点击暂停可停止检测

- [ ] **Step 4: 验证后台管理页**
- 切换到后台管理标签
- 显示分类统计汇总
- 显示完整日志表格
- 可以筛选类型
- 可以清空日志
