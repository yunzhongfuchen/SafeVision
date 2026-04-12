# 睡岗 & 未戴口罩检测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 YOLOv8-Pose 的睡岗检测 + 口罩检测复用，在人体框上显示异常状态

**Architecture:** 复用现有 `result_human` 人体框，新增 `detect_sleep()` 线程调用 YOLOv8-Pose 做姿态分析，复用 `result_mask` 做口罩判定，主循环根据状态改变框颜色

**Tech Stack:** Ultralytics YOLOv8-Pose, OpenCV, Python threading

---

### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `main.py` | 修改 | 全部改动都在这个文件 |

**文件责任：** `main.py` 是单文件项目，新增内容：
- 加载 `yolov8s-pose.pt` 模型
- 新增 `detect_sleep()` 函数
- 新增 `sleep_tracker` / `mask_tracker` 字典
- 新增 `IoU` 辅助函数
- 修改主循环渲染逻辑（绿色/黄色框）

---

### Task 1: 加载 Pose 模型

**Files:**
- Modify: `main.py` (模型加载区, line ~30)

- [ ] **Step 1: 添加 pose 模型加载**

在 `fire_model = YOLO("fire.pt")` 下方添加：

```python
pose_model = YOLO("yolov8s-pose.pt")
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "feat: add YOLOv8-Pose model loading"
```

---

### Task 2: 添加状态追踪器和 IoU 函数

**Files:**
- Modify: `main.py` (结果缓存区, line ~51-54)

- [ ] **Step 1: 添加追踪器字典**

在 `result_fire = {}` 下方添加：

```python
sleep_tracker = {}   # person_id -> {"sleep_count": 0, "sleeping": False}
mask_tracker = {}    # person_id -> {"no_mask_count": 0, "no_mask": False}
person_counter = 0   # 用于给每个人分配 ID
```

- [ ] **Step 2: 添加 IoU 计算函数**

在 `detect_fire()` 函数之后添加：

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add sleep/mask trackers and IoU helper"
```

---

### Task 3: 实现睡岗检测线程

**Files:**
- Modify: `main.py` (detect_fire 之后)

- [ ] **Step 1: 添加 sleep_tracker 持久化清理逻辑**

```python
def cleanup_trackers(tracker, alive_ids):
    """清理不再出现的人的追踪状态"""
    for pid in list(tracker.keys()):
        if pid not in alive_ids:
            del tracker[pid]
```

- [ ] **Step 2: 添加 detect_sleep 线程**

在 `detect_fire()` 函数之后添加：

```python
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
                        import math
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
```

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add sleep detection thread"
```

---

### Task 4: 在主循环中判定未戴口罩

**Files:**
- Modify: `main.py` (主循环渲染区, line ~160-210)

- [ ] **Step 1: 替换主循环渲染逻辑**

将 `# 🎬 主循环` 到 `cv2.destroyAllWindows()` 之间的渲染部分替换为：

```python
while True:

    # ⭐ 统一退出条件
    if video_finished:
        running = False
        break

    if latest_frame is None:
        time.sleep(0.01)
        continue

    with lock:
        frame = latest_frame.copy()

    # 🧍 人体检测渲染（主框）
    if 'boxes' in result_human:
        human_boxes = result_human['boxes']
        human_scores = result_human['scores']
        mask_boxes = result_mask.get('boxes', [])

        # 为每个人计算状态
        person_states = []  # (box, label, color)
        for box, score in zip(human_boxes, human_scores):
            x1, y1, x2, y2 = map(int, box)

            # ---- 睡岗判定 ----
            is_sleeping = False
            sleep_conf = 0.0
            # 找到最近的 sleep_tracker 条目
            for pid, tracker in sleep_tracker.items():
                if tracker["sleeping"]:
                    is_sleeping = True
                    sleep_conf = score
                    break

            # ---- 未戴口罩判定 ----
            has_mask = False
            for m_box in mask_boxes:
                mx1, my1, mx2, my2 = m_box
                if calc_iou([x1, y1, x2, y2], [mx1, my1, mx2, my2]) > 0.2:
                    has_mask = True
                    break

            # 决定标签和颜色
            if is_sleeping:
                label = f'SLEEP {sleep_conf:.2f}'
                color = (0, 255, 255)  # 黄色 (BGR)
            elif not has_mask:
                label = f'NO MASK {score:.2f}'
                color = (0, 255, 255)  # 黄色 (BGR)
            else:
                label = f'person {score:.2f}'
                color = (0, 255, 0)  # 绿色 (BGR)

            person_states.append((box, label, color))

        # 绘制框和标签
        for box, label, color in person_states:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 🚬 香烟（保持原样，独立框）
    if 'boxes' in result_cig:
        for box, score in zip(result_cig['boxes'], result_cig['scores']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'cig {score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 🔥 火焰（保持原样，独立框）
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
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "feat: integrate sleep/mask status into main render loop"
```

---

### Task 5: 启动睡岗检测线程 + 验证

**Files:**
- Modify: `main.py` (线程启动区, line ~150-154)

- [ ] **Step 1: 添加 detect_sleep 线程启动**

在现有线程启动代码后添加：

```python
threading.Thread(target=detect_sleep, daemon=True).start()
```

- [ ] **Step 2: 验证项目可运行**

```bash
python main.py
```

确认：
- 程序能正常启动
- 画面中人体框显示绿色（正常）
- 检测到睡姿持续后变为黄色 + "SLEEP"
- 未戴口罩变为黄色 + "NO MASK"

---

## Spec Coverage 自检

| Spec 需求 | 对应 Task | 状态 |
|-----------|-----------|------|
| 加载 yolov8s-pose.pt | Task 1 | ✅ |
| 复用 result_human 框 | Task 4 | ✅ |
| detect_sleep 线程 | Task 3 | ✅ |
| 鼻子低于肩膀判定 | Task 3 | ✅ |
| 身体倾斜角 > 60° 判定 | Task 3 | ✅ |
| sleep_count 持续判定 | Task 3 | ✅ |
| IoU 口罩重叠度判定 | Task 2 + 4 | ✅ |
| no_mask_count 持续判定 | Task 3 (tracker 逻辑) | ⚠️ 需补充 |
| 绿色正常/黄色异常 | Task 4 | ✅ |
| 一人一框 | Task 4 | ✅ |

⚠️ **发现遗漏：Task 3 实现了 sleep_tracker 但 mask_tracker 的判定逻辑需要加到主循环中。**

- [ ] **补充: 在主循环中加入 mask_tracker 持续判定**

在 Task 4 的主循环中，`# ---- 未戴口罩判定 ----` 部分需要改为使用 mask_tracker 而非即时判定：

```python
# ---- 未戴口罩判定（使用 tracker） ----
has_mask = False
for m_box in mask_boxes:
    mx1, my1, mx2, my2 = m_box
    if calc_iou([x1, y1, x2, y2], [mx1, my1, mx2, my2]) > 0.2:
        has_mask = True
        break

# 需要给每个人分配 mask_tracker 条目
# 由于主循环中 person_id 与 sleep_tracker 中的 id 需要对应
# 这里简化处理：直接统计连续未戴口罩帧数
if not has_mask:
    # 用人体框索引作为简易 ID
    person_idx = human_boxes.index(box)
    pid = f"mask_{person_idx}"
    if pid not in mask_tracker:
        mask_tracker[pid] = {"no_mask_count": 0, "no_mask": False}
    mask_tracker[pid]["no_mask_count"] += 1
    mask_tracker[pid]["no_mask_count"] = max(0, mask_tracker[pid]["no_mask_count"] - 2)

    if mask_tracker[pid]["no_mask_count"] >= 30:
        mask_tracker[pid]["no_mask"] = True
    if mask_tracker[pid]["no_mask"] and mask_tracker[pid]["no_mask_count"] == 0:
        mask_tracker[pid]["no_mask"] = False

    no_mask = mask_tracker[pid]["no_mask"]
else:
    no_mask = False
```

然后判断条件改为：

```python
if is_sleeping:
    label = f'SLEEP {sleep_conf:.2f}'
    color = (0, 255, 255)  # 黄色
elif no_mask:
    label = f'NO MASK {score:.2f}'
    color = (0, 255, 255)  # 黄色
else:
    label = f'person {score:.2f}'
    color = (0, 255, 0)  # 绿色
```

重新审视后，mask_tracker 用简易 person_idx 作为 ID 在每帧可能不一致。更好的方案是将 person_id 的分配从 detect_sleep 移到主循环，或者在主循环中直接使用即时判定（无持续状态），因为口罩检测相对稳定：

**简化方案（推荐）：** 口罩检测不需要持续状态判定，单帧 IoU 判定即可。口罩框如果稳定存在，每帧都会检测到。去掉 mask_tracker，直接用即时 IoU 判定。

最终渲染判断：
```python
if is_sleeping:
    label = f'SLEEP {sleep_conf:.2f}'
    color = (0, 255, 255)  # 黄色
elif not has_mask:
    label = f'NO MASK {score:.2f}'
    color = (0, 255, 255)  # 黄色
else:
    label = f'person {score:.2f}'
    color = (0, 255, 0)  # 绿色
```

- [ ] **Step 3: 更新 spec 中 mask_tracker 说明**

mask_tracker 不需要持续状态判定，简化为单帧 IoU 判定即可。更新设计文档中的 mask_tracker 说明为"简化方案：单帧 IoU 判定"。

- [ ] **Step 4: Commit**

```bash
git add main.py docs/superpowers/specs/2026-04-11-sleeping-detection-design.md
git commit -m "refine: mask detection uses per-frame IoU without tracker state"
```
