"""AI Video Surveillance Web UI - Gradio UI + Flask MJPEG Video Stream"""

import cv2
import time
import threading
import gradio as gr

# Flask imports
from flask import Flask, Response, request
import numpy as np


def _cors_headers():
    """Return CORS headers for cross-origin requests from Gradio UI."""
    origin = request.headers.get('Origin', '')
    if origin:
        return {'Access-Control-Allow-Origin': origin}
    return {}


def _add_cors(response):
    """Attach CORS headers to a Flask response."""
    origin = request.headers.get('Origin', '')
    if origin:
        response.headers['Access-Control-Allow-Origin'] = origin
    return response

import engine


# ======================
# Flask MJPEG Video Stream + JSON API endpoints
# ======================

flask_app = Flask(__name__)


@flask_app.after_request
def after_cors(response):
    return _add_cors(response)


@flask_app.route("/video_feed")
def video_feed():
    """MJPEG stream: serve pre-rendered JPEG bytes from engine._render_loop."""
    def generate():
        while True:
            jpeg = engine.annotated_jpeg
            if jpeg is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            else:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(img, "No video source", (150, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
                _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + enc.tobytes() + b"\r\n")
            time.sleep(0.033)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@flask_app.route("/api/stats")
def api_stats():
    """Return current detection stats as JSON."""
    stats = engine.get_stats()
    return {"stats": stats}


@flask_app.route("/api/logs")
def api_logs():
    """Return recent log entries as JSON."""
    entries = engine.get_log_entries(limit=10)
    type_map = {
        "fire": ("🔥 明火", "log-fire"),
        "smoke": ("🌫️ 烟雾", "log-smoke"),
        "cig": ("🚬 抽烟", "log-cig"),
        "no_mask": ("😷 未戴口罩", "log-no_mask"),
        "sleep": ("💤 睡岗", "log-sleep"),
        "uniform": ("🦺 未穿工服", "log-uniform"),
    }
    result = []
    for entry in entries:
        label, cls = type_map.get(entry.event_type, ("未知", ""))
        result.append({
            "time": entry.time,
            "label": label,
            "cls": cls,
            "score": entry.score,
            "snapshot": entry.snapshot,
        })
    return {"logs": result}


@flask_app.route("/api/fire-alert")
def api_fire_alert():
    """Return fire alert status as JSON."""
    from flask import jsonify
    has_fire = engine.get_stats()['fire'] > 0
    global fire_muted_until, fire_mute_lock
    with fire_mute_lock:
        muted = time.time() < fire_muted_until
    if muted:
        remaining = max(0, int(fire_muted_until - time.time()))
    else:
        remaining = 0
    return {"has_fire": has_fire, "muted": muted, "remaining": remaining}


@flask_app.route("/api/system-status")
def api_system_status():
    """Return system status as JSON."""
    status = engine.get_system_status()
    return {"status": status}


@flask_app.route("/api/admin-logs")
def api_admin_logs():
    """Return full admin logs as JSON."""
    entries = engine.get_log_entries(limit=100)
    type_map = {
        "fire": ("🔥 明火", "log-fire"),
        "smoke": ("🌫️ 烟雾", "log-smoke"),
        "cig": ("🚬 抽烟", "log-cig"),
        "no_mask": ("😷 未戴口罩", "log-no_mask"),
        "sleep": ("💤 睡岗", "log-sleep"),
        "uniform": ("🦺 未穿工服", "log-uniform"),
    }
    result = []
    for entry in entries:
        label, cls = type_map.get(entry.event_type, ("未知", ""))
        result.append({
            "time": entry.time,
            "label": label,
            "cls": cls,
            "score": entry.score,
            "snapshot": entry.snapshot,
        })
    return {"logs": result}


@flask_app.route("/api/clear-logs", methods=["POST"])
def api_clear_logs():
    """Clear all logs."""
    engine.clear_logs()
    return {"status": "ok"}


@flask_app.route("/api/export-csv")
def api_export_csv():
    """Export logs as a downloadable CSV file."""
    from flask import Response
    csv_data = engine.export_logs_csv()
    return Response(
        csv_data.encode("utf-8-sig"),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=detection_logs.csv"},
    )


@flask_app.route("/api/admin-stats")
def api_admin_stats():
    """Return admin page stat summary as JSON."""
    logs = engine.get_log_entries()
    counts = {"fire": 0, "smoke": 0, "cig": 0, "no_mask": 0, "sleep": 0, "uniform": 0}
    for entry in logs:
        counts[entry.event_type] = counts.get(entry.event_type, 0) + 1
    total = sum(counts.values())
    percentages = {}
    for key in counts:
        if total == 0:
            percentages[key] = "0%"
        else:
            percentages[key] = f"{counts[key]/total*100:.0f}%"
    return {"counts": counts, "percentages": percentages, "total": total}


@flask_app.route("/api/mute-fire", methods=["POST"])
def api_mute_fire():
    """Mute fire alarm for 30 seconds."""
    global fire_muted_until, fire_mute_lock
    with fire_mute_lock:
        fire_muted_until = time.time() + 30
    return {"status": "muted"}


def start_flask():
    """Run Flask in background thread"""
    flask_app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)


threading.Thread(target=start_flask, daemon=True).start()


# ======================
# Custom CSS for Light Theme
# ======================

CUSTOM_CSS = """
/* ============ Stat Cards ============ */
.stat-bar {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 10px;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    background: #fff;
    margin-bottom: 10px;
}
.stat-card {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 5px 10px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    white-space: nowrap;
    justify-content: center;
}
.stat-fire  { background: #fee2e2; color: #dc2626; }
.stat-smoke { background: #fef3c7; color: #d97706; }
.stat-cig   { background: #d1fae5; color: #059669; }
.stat-mask  { background: #dbeafe; color: #2563eb; }
.stat-sleep { background: #ede9fe; color: #7c3aed; }
.stat-uniform { background: #d1fae5; color: #059669; }

/* ============ Video Player ============ */
.video-wrapper {
    width: 100%;
    max-width: 100%;
    height: 600px;
    border: 2px solid #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
    background: #000;
}
.video-wrapper img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
    display: block;
}

/* ============ Controls Row ============ */
.controls-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 0 2px;
    flex-wrap: wrap;
}
.controls-row button {
    min-width: 80px;
}

/* ============ Hidden fire mute button ============ */
.hidden-btn { display: none !important; }

/* ============ Bottom row: file + status same line ============ */
.bottom-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 0px !important;
    width: 100% !important;
}
.bottom-row > div,
.bottom-row > .form {
    flex: 0 0 50% !important;
    width: 50% !important;
    min-width: 0 !important;
    max-width: 50% !important;
    margin: 0 !important;
    padding: 0 4px !important;
}
.bottom-row .file-component {
    width: 100% !important;
}
.bottom-row .file-component .wrap {
    height: 42px !important;
    min-height: 42px !important;
    max-height: 42px !important;
}
.bottom-row .file-component .file-preview-holder {
    display: none !important;
}
.bottom-row .file-component .icon-wrap {
    display: none !important;
}
.bottom-row .file-component .upload-button {
    height: 42px !important;
    line-height: 42px !important;
}
.bottom-row .file-component .file-info {
    display: none !important;
}

/* ============ Right Column ============ */
#monitor-right-col {
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
}

/* ============ Log Panel ============ */
#log-display-area {
    padding: 0 !important;
    margin: 0 !important;
}
/* Force zero padding on all Gradio wrapper elements */
#log-display-area .block,
#log-display-area .block > .wrap,
#log-display-area .wrap,
#log-display-area .form,
#log-display-area .form > div,
#log-display-area div:empty,
#log-display-area [data-testid="block"],
#log-display-area [data-testid="block"] .wrap {
    padding: 0 !important;
    margin: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}
.log-panel {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    height: 400px !important;
    overflow-y: auto;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    background: #fff;
}
/* Inner log entries spacing */
#log-panel-inner {
    padding: 10px !important;
}

/* ============ Admin Log Panel (fixed height with scroll) ============ */
.admin-log-panel {
    height: 480px;
    overflow-y: auto;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    background: #fff;
}
.log-entry { padding: 4px 8px; margin: 2px 0; border-radius: 4px; font-size: 12px; }
.log-fire { background: #fee2e2; color: #dc2626; border-left: 3px solid #dc2626; }
.log-smoke { background: #fef3c7; color: #d97706; border-left: 3px solid #d97706; }
.log-cig { background: #fef3c7; color: #d97706; border-left: 3px solid #d97706; }
.log-no_mask { background: #dbeafe; color: #2563eb; border-left: 3px solid #2563eb; }
.log-sleep { background: #ede9fe; color: #7c3aed; border-left: 3px solid #7c3aed; }
.log-uniform { background: #d1fae5; color: #059669; border-left: 3px solid #059669; }

/* ============ Fire Alert Indicator ============ */
.fire-alert {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 14px;
    min-height: 48px;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    background: #fff;
    margin-top: 8px;
}
.fire-dot {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #d1d5db;
}
.fire-dot.alert {
    background: #dc2626;
    animation: fire-flash 0.5s ease-in-out infinite alternate;
}
@keyframes fire-flash {
    from { opacity: 1; box-shadow: 0 0 6px #dc2626; }
    to   { opacity: 0.3; box-shadow: 0 0 2px #dc2626; }
}
.fire-label {
    font-size: 14px;
    font-weight: 600;
    color: #6b7280;
}
.fire-label.alert {
    color: #dc2626;
}
.fire-mute-btn {
    margin-left: auto;
    padding: 6px 14px;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    background: #f9fafb;
    cursor: pointer;
    font-size: 13px;
    color: #6b7280;
    white-space: nowrap;
}
.fire-mute-btn:hover {
    background: #e5e7eb;
}

/* ============ Page ============ */
.monitor-container { max-width: 1200px; margin: 0 auto; }
.gradio-container { background: #f5f7fa; }

/* ============ Snapshot Modal ============ */
.snap-btn {
    padding: 2px 8px;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    background: #f0f9ff;
    color: #0284c7;
    font-size: 11px;
    cursor: pointer;
    white-space: nowrap;
}
.snap-btn:hover { background: #e0f2fe; }
.snap-modal {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 9999;
}
.snap-backdrop {
    position: absolute;
    inset: 0;
    background: rgba(0,0,0,0.7);
}
.snap-image {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    max-width: 80vw;
    max-height: 80vh;
    border-radius: 8px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}
.snap-close {
    position: absolute;
    top: 20px; right: 30px;
    background: none;
    border: none;
    color: #fff;
    font-size: 28px;
    cursor: pointer;
    z-index: 10000;
}
"""

STATS_HTML = """
<div class="stat-bar">
    <span class="stat-card stat-fire">🔥 明火: <span id="stat-fire">0</span></span>
    <span class="stat-card stat-smoke">🌫️ 烟雾: <span id="stat-smoke">0</span></span>
    <span class="stat-card stat-cig">🚬 抽烟: <span id="stat-cig">0</span></span>
    <span class="stat-card stat-mask">😷 未戴: <span id="stat-mask">0</span></span>
    <span class="stat-card stat-sleep">💤 睡岗: <span id="stat-sleep">0</span></span>
    <span class="stat-card stat-uniform">🦺 工服: <span id="stat-uniform">0</span></span>
</div>
"""

LOG_HTML = """<div class="log-panel"><div id="log-panel-inner" style="padding:10px; color:#6b7280;">等待检测启动...</div></div>"""

VIDEO_HTML = """<div class="video-wrapper">
    <img id="video-feed" src="http://localhost:5000/video_feed" />
</div>"""

FIRE_ALERT_HTML = """
<div class="fire-alert" id="fire-alert-container">
    <div class="fire-dot" id="fire-dot"></div>
    <span class="fire-label" id="fire-label">明火状态：正常</span>
</div>
"""


# ======================
# Event Handlers
# ======================

fire_muted_until = 0.0
fire_mute_lock = threading.Lock()


def mute_fire():
    """Mute fire alarm for 30 seconds"""
    global fire_muted_until
    with fire_mute_lock:
        fire_muted_until = time.time() + 30


def is_fire_muted():
    """Check if fire alarm is currently muted"""
    with fire_mute_lock:
        return time.time() < fire_muted_until


def update_stats():
    """Update stat cards"""
    stats = engine.get_stats()
    return f"""
    <div class="stat-bar">
        <span class="stat-card stat-fire">🔥 明火: {stats['fire']}</span>
        <span class="stat-card stat-smoke">🌫️ 烟雾: {stats['smoke']}</span>
        <span class="stat-card stat-cig">🚬 抽烟: {stats['cig']}</span>
        <span class="stat-card stat-mask">😷 未戴: {stats['no_mask']}</span>
        <span class="stat-card stat-sleep">💤 睡岗: {stats['sleep']}</span>
        <span class="stat-card stat-uniform">🦺 工服: {stats['uniform']}</span>
    </div>
    """


def update_logs():
    """Update log stream (recent 10)"""
    entries = engine.get_log_entries(limit=10)
    if not entries:
        return '<div class="log-panel"><div style="padding:10px; color:#6b7280;">暂无异常记录</div></div>'

    type_map = {
        "fire": ("🔥 明火", "log-fire"),
        "smoke": ("🌫️ 烟雾", "log-smoke"),
        "cig": ("🚬 抽烟", "log-cig"),
        "no_mask": ("😷 未戴口罩", "log-no_mask"),
        "sleep": ("💤 睡岗", "log-sleep"),
        "uniform": ("🦺 未穿工服", "log-uniform"),
    }

    html = '<div class="log-panel"><div style="padding:8px;">'
    for idx, entry in enumerate(reversed(entries)):
        label, cls = type_map.get(entry.event_type, ("未知", ""))
        snap_btn = ""
        if entry.snapshot:
            snap_id = f"rsnap-{idx}"
            snap_btn = f'<button class="snap-btn" style="margin-left:4px;" onclick="showSnapshot(\'{snap_id}\')">📷</button>'
        html += f'<div class="log-entry {cls}" style="display:flex; justify-content:space-between; align-items:center;">'
        html += f'<span>{entry.time} {label} ({entry.score:.2f}){snap_btn}</span>'
        html += '</div>'
        if entry.snapshot:
            snap_id = f"rsnap-{idx}"
            html += f'<div id="{snap_id}" class="snap-modal" style="display:none;"><div class="snap-backdrop" onclick="closeSnapshot()"></div><img src="{entry.snapshot}" class="snap-image" /><button class="snap-close" onclick="closeSnapshot()">✕</button></div>'
    html += '</div></div>'
    return html


def update_fire_alert():
    """Update fire alert indicator"""
    has_fire = engine.get_stats()['fire'] > 0
    muted = is_fire_muted()

    if has_fire and not muted:
        return """
<div class="fire-alert" id="fire-alert-container">
    <div class="fire-dot alert"></div>
    <span class="fire-label alert">🔥 检测到明火！</span>
    <button class="fire-mute-btn" id="fire-mute-btn">关闭警报</button>
</div>
<script>
(function() {
    var btn = document.getElementById('fire-mute-btn');
    if (btn) {
        btn.addEventListener('click', function(e) {
            e.preventDefault(); e.stopPropagation();
            var hiddenBtn = document.querySelector('.hidden-btn button');
            if (hiddenBtn) hiddenBtn.click();
        });
    }
})();
</script>
"""
    elif muted:
        remaining = max(0, int(fire_muted_until - time.time()))
        return f"""
<div class="fire-alert" id="fire-alert-container">
    <div class="fire-dot alert"></div>
    <span class="fire-label alert">🔥 明火检测中（已静音 {remaining}s）</span>
</div>
"""
    else:
        return FIRE_ALERT_HTML


def play_file(filepath):
    """Play video file"""
    if not filepath:
        return "请先选择视频文件"
    result = engine.start_detection("file", filepath)
    return result


def play_camera():
    """Start camera"""
    return engine.start_detection("camera")


def pause():
    """Pause/Stop"""
    return engine.stop_detection()


def update_system_status():
    """Update system status card"""
    status = engine.get_system_status()
    run_label = "🟢 运行中" if status["running"] else "⚪ 已停止"
    run_color = "#16a34a" if status["running"] else "#6b7280"

    threads = status.get("threads", {})
    thread_names = {
        "detect_cig": "抽烟检测",
        "detect_mask": "口罩检测",
        "detect_fire": "明火/烟雾",
        "detect_sleep": "睡岗检测",
        "detect_uniform": "工服检测",
        "_render_loop": "渲染输出",
    }
    thread_html = ""
    for key, label in thread_names.items():
        info = threads.get(key, {})
        running_str = "运行中" if info.get("running") else "—"
        fps_str = f"{info.get('fps', 0):.1f} fps" if info.get("running") else ""
        thread_html += f'<div style="display:flex; justify-content:space-between; padding:3px 0; font-size:12px;">'
        thread_html += f'<span>{label}</span>'
        thread_html += f'<span style="color:#6b7280;">{running_str} {fps_str}</span>'
        thread_html += f'</div>'

    stats = engine.get_stats()
    total_detections = sum(stats.values())
    return f"""
    <div style="padding:8px;">
        <div style="display:flex; justify-content:space-between; padding:4px 0; font-size:13px; font-weight:600;">
            <span>运行状态</span><span style="color:{run_color};">{run_label}</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding:4px 0; font-size:13px;">
            <span>视频源</span><span>{status['source']}</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding:4px 0; font-size:13px;">
            <span>运行时长</span><span>{status['uptime']}</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding:4px 0; font-size:13px;">
            <span>视频帧率</span><span>{status['video_fps']} fps</span>
        </div>
        <div style="border-top:1px solid #e5e7eb; margin:6px 0;"></div>
        <div style="font-size:13px; font-weight:600; margin-bottom:4px;">检测线程</div>
        {thread_html}
        <div style="border-top:1px solid #e5e7eb; margin:6px 0;"></div>
        <div style="font-size:13px; font-weight:600; margin-bottom:4px;">当前帧目标数</div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:4px; font-size:12px;">
            <span>🔥 明火: {stats['fire']}</span>
            <span>🌫️ 烟雾: {stats['smoke']}</span>
            <span>🚬 抽烟: {stats['cig']}</span>
            <span>😷 未戴: {stats['no_mask']}</span>
            <span>💤 睡岗: {stats['sleep']}</span>
            <span>🦺 工服: {stats['uniform']}</span>
        </div>
    </div>
    """


def update_admin_stats():
    """Update admin page stat summary (with no_mask)"""
    logs = engine.get_log_entries()
    counts = {"fire": 0, "smoke": 0, "cig": 0, "no_mask": 0, "sleep": 0, "uniform": 0}
    for entry in logs:
        counts[entry.event_type] = counts.get(entry.event_type, 0) + 1
    total = sum(counts.values())

    def pct(key):
        if total == 0:
            return "0%"
        return f"{counts[key]/total*100:.0f}%"

    return f"""
    <div style="display:flex; gap:8px; justify-content:center; padding:10px; flex-wrap:wrap;">
        <div style="background:#fee2e2; padding:10px 16px; border-radius:8px; text-align:center; flex:1; min-width:80px;">
            <div style="font-size:20px; color:#dc2626; font-weight:700;">{counts['fire']}</div>
            <div style="font-size:10px; color:#991b1b;">明火 ({pct('fire')})</div>
        </div>
        <div style="background:#fef3c7; padding:10px 16px; border-radius:8px; text-align:center; flex:1; min-width:80px;">
            <div style="font-size:20px; color:#d97706; font-weight:700;">{counts['smoke']}</div>
            <div style="font-size:10px; color:#92400e;">烟雾 ({pct('smoke')})</div>
        </div>
        <div style="background:#d1fae5; padding:10px 16px; border-radius:8px; text-align:center; flex:1; min-width:80px;">
            <div style="font-size:20px; color:#059669; font-weight:700;">{counts['cig']}</div>
            <div style="font-size:10px; color:#065f46;">抽烟 ({pct('cig')})</div>
        </div>
        <div style="background:#dbeafe; padding:10px 16px; border-radius:8px; text-align:center; flex:1; min-width:80px;">
            <div style="font-size:20px; color:#2563eb; font-weight:700;">{counts['no_mask']}</div>
            <div style="font-size:10px; color:#1e40af;">未戴 ({pct('no_mask')})</div>
        </div>
        <div style="background:#ede9fe; padding:10px 16px; border-radius:8px; text-align:center; flex:1; min-width:80px;">
            <div style="font-size:20px; color:#7c3aed; font-weight:700;">{counts['sleep']}</div>
            <div style="font-size:10px; color:#5b21b6;">睡岗 ({pct('sleep')})</div>
        </div>
        <div style="background:#d1fae5; padding:10px 16px; border-radius:8px; text-align:center; flex:1; min-width:80px;">
            <div style="font-size:20px; color:#059669; font-weight:700;">{counts['uniform']}</div>
            <div style="font-size:10px; color:#065f46;">工服 ({pct('uniform')})</div>
        </div>
    </div>
    """


def update_admin_logs(filter_type="全部"):
    """Update admin page full log with filter, limited to 100 entries"""
    entries = engine.get_log_entries(limit=100)
    if not entries:
        return '<div class="admin-log-panel"><div style="padding:10px; color:#6b7280;">暂无日志</div></div>'

    type_map = {
        "fire": ("🔥 明火", "log-fire"),
        "smoke": ("🌫️ 烟雾", "log-smoke"),
        "cig": ("🚬 抽烟", "log-cig"),
        "no_mask": ("😷 未戴口罩", "log-no_mask"),
        "sleep": ("💤 睡岗", "log-sleep"),
        "uniform": ("🦺 未穿工服", "log-uniform"),
    }

    type_filter = None
    for k, (label, _) in type_map.items():
        if label == filter_type:
            type_filter = k
            break

    html = '<div class="admin-log-panel"><div style="padding:4px; position:relative;">'
    html += '<div style="display:grid; grid-template-columns:80px 1fr 80px 80px; gap:4px; color:#6b7280; font-size:11px; padding:0 8px; margin-bottom:6px; position:sticky; top:0; background:#fff; z-index:1;">'
    html += '<span>时间</span><span>类型</span><span>置信度</span><span></span>'
    html += '</div>'

    for idx, entry in enumerate(reversed(entries)):
        if type_filter and entry.event_type != type_filter:
            continue
        label, cls = type_map.get(entry.event_type, ("未知", ""))
        snap_btn = ""
        if entry.snapshot:
            snap_id = f"snap-{idx}"
            snap_btn = f'<button class="snap-btn" data-snap="{snap_id}" onclick="showSnapshot(\'{snap_id}\')">查看截图</button>'
        html += f'<div class="log-entry {cls}" style="display:grid; grid-template-columns:80px 1fr 80px 80px; gap:4px; align-items:center;">'
        html += f'<span>{entry.time}</span><span>{label}</span><span>{entry.score:.2f}</span><span>{snap_btn}</span>'
        html += '</div>'
        if entry.snapshot:
            snap_id = f"snap-{idx}"
            html += f'<div id="{snap_id}" class="snap-modal" style="display:none;"><div class="snap-backdrop" onclick="closeSnapshot()"></div><img src="{entry.snapshot}" class="snap-image" /><button class="snap-close" onclick="closeSnapshot()">✕</button></div>'
    html += '</div></div>'
    return html


def do_clear_logs():
    engine.clear_logs()
    return '<div class="admin-log-panel"><div style="padding:10px; color:#6b7280;">日志已清空</div></div>'


def export_logs():
    """Export logs as CSV file"""
    csv_data = engine.export_logs_csv()
    import tempfile
    import os
    tmpdir = tempfile.gettempdir()
    path = os.path.join(tmpdir, "detection_logs.csv")
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(csv_data)
    return f"已导出到 {path}"


def get_config_values():
    """Return current config values for UI"""
    cfg = engine.get_config()
    alerts = engine.get_alerts()
    display = engine.get_display()
    return (
        cfg["conf_fire"],
        cfg["conf_cig"],
        cfg["conf_mask"],
        cfg["conf_pose"],
        cfg["conf_uniform"],
        cfg["frames_fire"],
        cfg["frames_cig"],
        cfg["frames_mask"],
        cfg["frames_sleep"],
        cfg["frames_uniform"],
        cfg["log_cooldown"],
        alerts["fire"],
        alerts["smoke"],
        alerts["cig"],
        alerts["no_mask"],
        alerts["sleep"],
        alerts["uniform"],
        display.get("fire", False),
        display.get("smoke", False),
        display.get("cig", False),
        display.get("no_mask", False),
        display.get("sleep", False),
        display.get("uniform", False),
    )


def apply_config(conf_fire, conf_cig, conf_mask, conf_pose, conf_uniform,
                 frames_fire, frames_cig, frames_mask, frames_sleep, frames_uniform,
                 log_cooldown,
                 alert_fire, alert_smoke, alert_cig, alert_no_mask, alert_sleep, alert_uniform,
                 display_fire, display_smoke, display_cig, display_no_mask, display_sleep, display_uniform):
    """Apply configuration from admin page"""
    engine.set_config("conf_fire", conf_fire)
    engine.set_config("conf_cig", conf_cig)
    engine.set_config("conf_mask", conf_mask)
    engine.set_config("conf_pose", conf_pose)
    engine.set_config("conf_uniform", conf_uniform)
    engine.set_config("frames_fire", int(frames_fire))
    engine.set_config("frames_cig", int(frames_cig))
    engine.set_config("frames_mask", int(frames_mask))
    engine.set_config("frames_sleep", int(frames_sleep))
    engine.set_config("frames_uniform", int(frames_uniform))
    engine.set_config("log_cooldown", log_cooldown)
    engine.set_alert("fire", alert_fire)
    engine.set_alert("smoke", alert_smoke)
    engine.set_alert("cig", alert_cig)
    engine.set_alert("no_mask", alert_no_mask)
    engine.set_alert("sleep", alert_sleep)
    engine.set_alert("uniform", alert_uniform)
    engine.set_display("fire", display_fire)
    engine.set_display("smoke", display_smoke)
    engine.set_display("cig", display_cig)
    engine.set_display("no_mask", display_no_mask)
    engine.set_display("sleep", display_sleep)
    engine.set_display("uniform", display_uniform)
    return "✅ 配置已应用"


# ======================
# Gradio Blocks UI
# ======================

with gr.Blocks(title="AI 视频监控", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI 视频监控系统")

    # Hidden fire mute toggle button (CSS hides it)
    fire_mute_btn = gr.Button(elem_classes=["hidden-btn"])

    # State: log filter stored in JS (window.currentAdminLogFilter)

    with gr.Tabs():
        with gr.TabItem("实时监控"):
            with gr.Row():
                # Left: Video Player + Controls
                with gr.Column(scale=3):
                    video_output = gr.HTML(VIDEO_HTML)
                    # Row 1: control buttons
                    with gr.Row(elem_classes=["controls-row"]):
                        play_btn = gr.Button("▶ 播放", variant="primary")
                        pause_btn = gr.Button("⏸ 暂停", variant="secondary")
                        camera_btn = gr.Button("📷 摄像头", variant="secondary")
                    # Row 2: file select + status on same line (50/50)
                    with gr.Row(elem_classes=["bottom-row"]):
                        file_input = gr.File(
                            label="",
                            file_types=[".mp4", ".avi", ".mkv"],
                            type="filepath",
                        )
                        status_display = gr.Textbox(label="状态", interactive=False)

                # Right: Stats + Log + Fire Alert
                with gr.Column(scale=1, elem_id="monitor-right-col"):
                    stats_display = gr.HTML(STATS_HTML)
                    log_display = gr.HTML(LOG_HTML, elem_id="log-display-area", elem_classes=["log-area"])
                    fire_alert = gr.HTML(FIRE_ALERT_HTML)

        with gr.TabItem("后台管理"):
            with gr.Tabs():
                # Sub-tab 1: Model Config
                with gr.TabItem("模型配置"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**▸ 阈值设定**")
                            conf_fire = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="明火/烟雾检测阈值")
                            conf_cig = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="抽烟检测阈值")
                            conf_mask = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="口罩检测阈值")
                            conf_pose = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="睡岗检测阈值")
                            conf_uniform = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="工服检测阈值")

                            gr.Markdown("**▸ 告警开关**")
                            alert_fire = gr.Checkbox(label="明火", value=True)
                            alert_smoke = gr.Checkbox(label="烟雾", value=True)
                            alert_cig = gr.Checkbox(label="抽烟", value=True)
                            alert_no_mask = gr.Checkbox(label="未戴口罩", value=True)
                            alert_sleep = gr.Checkbox(label="睡岗", value=True)
                            alert_uniform = gr.Checkbox(label="工服", value=True)

                        with gr.Column():
                            gr.Markdown("**▸ 判定帧数**")
                            frames_fire = gr.Slider(1, 30, value=5, step=1, label="明火/烟雾判定帧数")
                            frames_cig = gr.Slider(1, 30, value=5, step=1, label="抽烟判定帧数")
                            frames_mask = gr.Slider(1, 30, value=5, step=1, label="口罩判定帧数")
                            frames_sleep = gr.Slider(30, 300, value=150, step=10, label="睡岗判定帧数")
                            frames_uniform = gr.Slider(1, 300, value=20, step=1, label="工服判定帧数")

                            gr.Markdown("**▸ 检测框显示**")
                            display_fire = gr.Checkbox(label="明火", value=False)
                            display_smoke = gr.Checkbox(label="烟雾", value=False)
                            display_cig = gr.Checkbox(label="抽烟", value=False)
                            display_no_mask = gr.Checkbox(label="未戴口罩", value=False)
                            display_sleep = gr.Checkbox(label="睡岗", value=False)
                            display_uniform = gr.Checkbox(label="工服", value=False)
                    apply_config_btn = gr.Button("✅ 应用配置", variant="primary")
                    config_status = gr.Textbox(label="", interactive=False, visible=False)
                    log_cooldown = gr.Slider(1.0, 15.0, value=5.0, step=0.5, label="日志冷却时间(秒)", visible=False)

                # Sub-tab 2: Log Management
                with gr.TabItem("日志管理"):
                    admin_stats = gr.HTML(
                        """<div id="admin-stats" style="text-align:center; color:#6b7280;">等待数据...</div>"""
                    )
                    with gr.Row():
                        log_filter = gr.Dropdown(
                            choices=["全部", "🔥 明火", "🌫️ 烟雾", "🚬 抽烟", "😷 未戴口罩", "💤 睡岗", "🦺 未穿工服"],
                            value="全部",
                            label="筛选类型",
                            scale=1,
                        )
                        export_btn = gr.Button("📥 导出CSV", variant="secondary", scale=1)
                        clear_log_btn = gr.Button("🗑️ 清空日志", variant="secondary", scale=1)
                    admin_log = gr.HTML(
                        """<div class="admin-log-panel"><div id="admin-log-inner" style="padding:10px; color:#6b7280;">暂无日志</div></div>"""
                    )

                # Sub-tab 3: System Status
                with gr.TabItem("系统状态"):
                    admin_status = gr.HTML(
                        """<div id="admin-status" style="text-align:center; color:#6b7280; padding:20px;">等待数据...</div>"""
                    )

            # ======================
            # Event Bindings
            # ======================

            play_btn.click(play_file, inputs=[file_input], outputs=[status_display])
            camera_btn.click(play_camera, outputs=[status_display])
            pause_btn.click(pause, outputs=[status_display])
            # Hidden fire mute button: toggles state
            fire_mute_btn.click(mute_fire)

            # Admin page: polling handled by JS setInterval below

            # Filter dropdown: update JS variable and refresh admin logs via JS
            log_filter.change(None, inputs=[log_filter], js="""
            function(v) {
                window.currentAdminLogFilter = v;
                window.lastAdminLogCount = 0;
                if (window.updateAdminLogs) window.updateAdminLogs();
                return v;
            }
            """)
            clear_log_btn.click(None, js="""
            function() {
                fetch('http://' + window.location.hostname + ':5000/api/clear-logs', {method: 'POST'}).then(function(){
                    window.lastAdminLogCount = -1;
                    if (window.updateAdminLogs) window.updateAdminLogs();
                });
            }
            """)

            # Config events
            apply_config_btn.click(
                apply_config,
                inputs=[conf_fire, conf_cig, conf_mask, conf_pose, conf_uniform,
                        frames_fire, frames_cig, frames_mask, frames_sleep, frames_uniform,
                        log_cooldown,
                        alert_fire, alert_smoke, alert_cig, alert_no_mask, alert_sleep, alert_uniform,
                        display_fire, display_smoke, display_cig, display_no_mask, display_sleep, display_uniform],
                outputs=[config_status],
            )
            export_btn.click(None, js="""
            function() {
                var url = 'http://' + window.location.hostname + ':5000/api/export-csv';
                var a = document.createElement('a');
                a.href = url;
                a.download = 'detection_logs.csv';
                a.target = '_blank';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }
            """)

    # Snapshot JS + polling for real-time stats/logs/fire-alert
    demo.load(None, js="""
function() {
    window.showSnapshot = function(id) {
        var el = document.getElementById(id);
        if (el) el.style.display = 'block';
    };
    window.closeSnapshot = function() {
        var modals = document.querySelectorAll('.snap-modal');
        modals.forEach(function(m) { m.style.display = 'none'; });
    };

    /* ---- Fix log panel width: inject styles into all shadow roots ---- */
    function injectLogPanelStyles() {
        var css = '#log-display-area .block,#log-display-area .wrap,#log-display-area .form,#log-display-area .wrap > div,#log-display-area [data-testid="block"]{padding:0!important;margin:0!important;width:100%!important;box-sizing:border-box!important;}';
        // Inject into light DOM
        var style = document.createElement('style');
        style.textContent = css;
        document.head.appendChild(style);
        // Also inject into shadow roots
        function injectIntoShadow(root) {
            var s = root.querySelector('#log-panel-fix-style');
            if (!s) {
                s = document.createElement('style');
                s.id = 'log-panel-fix-style';
                s.textContent = css;
                root.appendChild(s);
            }
        }
        // Check app root shadow
        var appRoots = document.querySelectorAll('gradio-app');
        appRoots.forEach(function(el) {
            if (el.shadowRoot) injectIntoShadow(el.shadowRoot);
        });
        // Check all shadow roots recursively
        var walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
        while (walker.nextNode()) {
            var node = walker.currentNode;
            if (node.shadowRoot) injectIntoShadow(node.shadowRoot);
        }
    }
    setTimeout(injectLogPanelStyles, 500);
    setTimeout(injectLogPanelStyles, 2000);
    setTimeout(injectLogPanelStyles, 5000);

    /* ---- Polling: update stats, logs, fire alert without full page refresh ---- */
    /* ---- API base: Flask runs on port 5000, Gradio on 7860 ---- */
    var API = 'http://' + window.location.hostname + ':5000';

    /* ---- Deep element lookup including shadow roots ---- */
    function deepFindElement(id) {
        var el = document.getElementById(id);
        if (el) return el;
        // Search inside all shadow roots
        var all = document.querySelectorAll('*');
        for (var i = 0; i < all.length; i++) {
            if (all[i].shadowRoot) {
                var found = all[i].shadowRoot.getElementById(id);
                if (found) return found;
            }
        }
        return null;
    }

    function updateStats() {
        fetch(API + '/api/stats').then(function(r){return r.json();}).then(function(d){
            var s = d.stats;
            var el = document.getElementById('stat-fire');
            if (el) el.textContent = s.fire;
            el = document.getElementById('stat-smoke');
            if (el) el.textContent = s.smoke;
            el = document.getElementById('stat-cig');
            if (el) el.textContent = s.cig;
            el = document.getElementById('stat-mask');
            if (el) el.textContent = s.no_mask;
            el = document.getElementById('stat-sleep');
            if (el) el.textContent = s.sleep;
            el = document.getElementById('stat-uniform');
            if (el) el.textContent = s.uniform;
        }).catch(function(){});
    }

    var lastLogKey = '';
    function updateLogs() {
        fetch(API + '/api/logs').then(function(r){return r.json();}).then(function(d){
            var logs = d.logs;
            if (logs.length === 0) {
                var panel = deepFindElement('log-panel-inner');
                if (panel) panel.innerHTML = '<div style="padding:10px; color:#6b7280;">暂无异常记录</div>';
                return;
            }
            var key = logs[logs.length - 1].time + '|' + logs.length;
            if (key === lastLogKey) return; // no new entries
            lastLogKey = key;
            var panel = deepFindElement('log-panel-inner');
            if (!panel) {
                return;
            }
            if (logs.length === 0) {
                panel.innerHTML = '<div style="padding:10px; color:#6b7280;">暂无异常记录</div>';
                return;
            }
            var html = '';
            for (var i = logs.length - 1; i >= 0; i--) {
                var entry = logs[i];
                var idx = logs.length - 1 - i;
                html += '<div class="log-entry ' + entry.cls + '" style="display:flex; justify-content:space-between; align-items:center;">';
                html += '<span>' + entry.time + ' ' + entry.label + ' (' + entry.score.toFixed(2) + ')';
                if (entry.snapshot) {
                    var snapId = 'rsnap-' + idx;
                    html += '<button class="snap-btn" style="margin-left:4px;" onclick="showSnapshot(\\'' + snapId + '\\')">📷</button>';
                }
                html += '</span></div>';
                if (entry.snapshot) {
                    var snapId2 = 'rsnap-' + idx;
                    html += '<div id="' + snapId2 + '" class="snap-modal" style="display:none;"><div class="snap-backdrop" onclick="closeSnapshot()"></div><img src="' + entry.snapshot + '" class="snap-image" /><button class="snap-close" onclick="closeSnapshot()">✕</button></div>';
                }
            }
            panel.innerHTML = html;
            console.log('[updateLogs] panel updated successfully');
        }).catch(function(e){
            console.error('[updateLogs] fetch error:', e);
        });
    }

    function updateFireAlert() {
        fetch(API + '/api/fire-alert').then(function(r){return r.json();}).then(function(d){
            var dot = document.getElementById('fire-dot');
            var label = document.getElementById('fire-label');
            var container = document.getElementById('fire-alert-container');
            if (!dot || !label) return;
            if (d.has_fire && !d.muted) {
                dot.className = 'fire-dot alert';
                label.className = 'fire-label alert';
                label.textContent = '🔥 检测到明火！';
                // Add mute button if not present
                if (!document.getElementById('fire-mute-inline-btn')) {
                    var btn = document.createElement('button');
                    btn.className = 'fire-mute-btn';
                    btn.id = 'fire-mute-inline-btn';
                    btn.textContent = '关闭警报';
                    btn.addEventListener('click', function(e) {
                        e.preventDefault(); e.stopPropagation();
                        fetch(API + '/api/mute-fire', {method:'POST'}).then(function(){updateFireAlert();});
                    });
                    container.appendChild(btn);
                }
            } else if (d.muted) {
                dot.className = 'fire-dot alert';
                label.className = 'fire-label alert';
                label.textContent = '🔥 明火检测中（已静音 ' + d.remaining + 's）';
                var existingBtn = document.getElementById('fire-mute-inline-btn');
                if (existingBtn) existingBtn.remove();
            } else {
                dot.className = 'fire-dot';
                label.className = 'fire-label';
                label.textContent = '明火状态：正常';
                var existingBtn = document.getElementById('fire-mute-inline-btn');
                if (existingBtn) existingBtn.remove();
            }
        }).catch(function(){});
    }

    /* ---- Polling: update admin page data (every 5s) ---- */
    function updateAdminStatus() {
        fetch(API + '/api/system-status').then(function(r){return r.json();}).then(function(d){
            var el = document.getElementById('admin-status');
            if (!el) return;
            var s = d.status;
            var runLabel = s.running ? '🟢 运行中' : '⚪ 已停止';
            var runColor = s.running ? '#16a34a' : '#6b7280';
            var threads = s.threads || {};
            var threadNames = {
                'detect_cig': '抽烟检测',
                'detect_mask': '口罩检测', 'detect_fire': '明火/烟雾',
                'detect_sleep': '睡岗检测', 'detect_uniform': '工服检测',
                '_render_loop': '渲染输出'
            };
            var threadHtml = '';
            for (var key in threadNames) {
                var info = threads[key] || {};
                var runningStr = info.running ? '运行中' : '—';
                var fpsStr = info.running ? info.fps.toFixed(1) + ' fps' : '';
                threadHtml += '<div style="display:flex; justify-content:space-between; padding:3px 0; font-size:12px;">';
                threadHtml += '<span>' + threadNames[key] + '</span>';
                threadHtml += '<span style="color:#6b7280;">' + runningStr + ' ' + fpsStr + '</span>';
                threadHtml += '</div>';
            }
            var statsHtml = '<div style="display:grid; grid-template-columns:1fr 1fr; gap:4px; font-size:12px;">';
            if (s.stats) {
                statsHtml += '<span>🔥 明火: ' + s.stats.fire + '</span>';
                statsHtml += '<span>🌫️ 烟雾: ' + s.stats.smoke + '</span>';
                statsHtml += '<span>🚬 抽烟: ' + s.stats.cig + '</span>';
                statsHtml += '<span>😷 未戴: ' + s.stats.no_mask + '</span>';
                statsHtml += '<span>💤 睡岗: ' + s.stats.sleep + '</span>';
            }
            statsHtml += '</div>';
            el.innerHTML = '<div style="padding:8px;">' +
                '<div style="display:flex; justify-content:space-between; padding:4px 0; font-size:13px; font-weight:600;">' +
                '<span>运行状态</span><span style="color:' + runColor + ';">' + runLabel + '</span></div>' +
                '<div style="display:flex; justify-content:space-between; padding:4px 0; font-size:13px;">' +
                '<span>视频源</span><span>' + s.source + '</span></div>' +
                '<div style="display:flex; justify-content:space-between; padding:4px 0; font-size:13px;">' +
                '<span>运行时长</span><span>' + s.uptime + '</span></div>' +
                '<div style="display:flex; justify-content:space-between; padding:4px 0; font-size:13px;">' +
                '<span>视频帧率</span><span>' + s.video_fps + ' fps</span></div>' +
                '<div style="border-top:1px solid #e5e7eb; margin:6px 0;"></div>' +
                '<div style="font-size:13px; font-weight:600; margin-bottom:4px;">检测线程</div>' +
                threadHtml +
                '<div style="border-top:1px solid #e5e7eb; margin:6px 0;"></div>' +
                '<div style="font-size:13px; font-weight:600; margin-bottom:4px;">当前帧目标数</div>' +
                statsHtml + '</div>';
        }).catch(function(){});
    }

    function updateAdminStats() {
        fetch(API + '/api/admin-stats').then(function(r){return r.json();}).then(function(d){
            var el = document.getElementById('admin-stats');
            if (!el) return;
            var c = d.counts;
            var p = d.percentages;
            var html = '<div style="display:flex; gap:8px; justify-content:center; padding:10px; flex-wrap:wrap;">';
            html += '<div style="background:#fee2e2; padding:10px 16px; border-radius:8px; text-align:center; flex:1; min-width:80px;">';
            html += '<div style="font-size:20px; color:#dc2626; font-weight:700;">' + c.fire + '</div>';
            html += '<div style="font-size:10px; color:#991b1b;">明火 (' + p.fire + ')</div></div>';
            html += '<div style="background:#fef3c7; padding:10px 16px; border-radius:8px; text-align:center; flex:1; min-width:80px;">';
            html += '<div style="font-size:20px; color:#d97706; font-weight:700;">' + c.smoke + '</div>';
            html += '<div style="font-size:10px; color:#92400e;">烟雾 (' + p.smoke + ')</div></div>';
            html += '<div style="background:#d1fae5; padding:10px 16px; border-radius:8px; text-align:center; flex:1; min-width:80px;">';
            html += '<div style="font-size:20px; color:#059669; font-weight:700;">' + c.cig + '</div>';
            html += '<div style="font-size:10px; color:#065f46;">抽烟 (' + p.cig + ')</div></div>';
            html += '<div style="background:#dbeafe; padding:10px 16px; border-radius:8px; text-align:center; flex:1; min-width:80px;">';
            html += '<div style="font-size:20px; color:#2563eb; font-weight:700;">' + c.no_mask + '</div>';
            html += '<div style="font-size:10px; color:#1e40af;">未戴 (' + p.no_mask + ')</div></div>';
            html += '<div style="background:#ede9fe; padding:10px 16px; border-radius:8px; text-align:center; flex:1; min-width:80px;">';
            html += '<div style="font-size:20px; color:#7c3aed; font-weight:700;">' + c.sleep + '</div>';
            html += '<div style="font-size:10px; color:#5b21b6;">睡岗 (' + p.sleep + ')</div></div>';
            html += '<div style="background:#d1fae5; padding:10px 16px; border-radius:8px; text-align:center; flex:1; min-width:80px;">';
            html += '<div style="font-size:20px; color:#059669; font-weight:700;">' + c.uniform + '</div>';
            html += '<div style="font-size:10px; color:#065f46;">工服 (' + p.uniform + ')</div></div>';
            html += '</div>';
            el.innerHTML = html;
        }).catch(function(){});
    }

    window.lastAdminLogCount = -1;  // -1 ensures first run always updates
    window.currentAdminLogFilter = '全部';
    window.updateAdminLogs = function() {
        var filter = window.currentAdminLogFilter;
        fetch(API + '/api/admin-logs').then(function(r){return r.json();}).then(function(d){
            var logs = d.logs;
            if (logs.length === lastAdminLogCount) return;
            lastAdminLogCount = logs.length;
            var el = document.getElementById('admin-log-inner');
            if (!el) return;
            var html = '<div style="display:grid; grid-template-columns:80px 1fr 80px 80px; gap:4px; color:#6b7280; font-size:11px; padding:0 8px; margin-bottom:6px;">';
            html += '<span>时间</span><span>类型</span><span>置信度</span><span></span></div>';
            for (var i = logs.length - 1; i >= 0; i--) {
                var entry = logs[i];
                var idx = logs.length - 1 - i;
                if (filter !== '全部' && entry.label !== filter) continue;
                var snapBtn = '';
                if (entry.snapshot) {
                    var snapId = 'snap-' + idx;
                    snapBtn = '<button class="snap-btn" data-snap="' + snapId + '" onclick="showSnapshot(\\'' + snapId + '\\')">查看截图</button>';
                }
                html += '<div class="log-entry ' + entry.cls + '" style="display:grid; grid-template-columns:80px 1fr 80px 80px; gap:4px; align-items:center;">';
                html += '<span>' + entry.time + '</span><span>' + entry.label + '</span><span>' + entry.score.toFixed(2) + '</span><span>' + snapBtn + '</span>';
                html += '</div>';
                if (entry.snapshot) {
                    var snapId2 = 'snap-' + idx;
                    html += '<div id="' + snapId2 + '" class="snap-modal" style="display:none;"><div class="snap-backdrop" onclick="closeSnapshot()"></div><img src="' + entry.snapshot + '" class="snap-image" /><button class="snap-close" onclick="closeSnapshot()">✕</button></div>';
                }
            }
            el.innerHTML = html;
        }).catch(function(){});
    }

    // Poll every 2 seconds for main page data
    setInterval(function() {
        updateStats();
        updateLogs();
        updateFireAlert();
    }, 2000);

    // Poll every 5 seconds for admin page data
    setInterval(function() {
        updateAdminStatus();
        updateAdminStats();
        updateAdminLogs();
    }, 5000);

    // Initial fetch after a short delay
    setTimeout(function() {
        updateStats();
        updateLogs();
        updateFireAlert();
        updateAdminStatus();
        updateAdminStats();
        updateAdminLogs();
    }, 500);
}
""")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
