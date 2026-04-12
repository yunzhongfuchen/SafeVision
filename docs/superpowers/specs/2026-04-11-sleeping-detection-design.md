# 睡岗 & 未戴口罩检测设计文档

**日期:** 2026-04-11
**主题:** 基于 YOLOv8-Pose 的睡岗检测 + 口罩检测复用

## 需求

在现有 AI 视频监控系统中增加：
1. 睡岗检测 — 当人员处于趴桌、低头等睡眠姿态并持续一段时间后标记
2. 未戴口罩检测 — 当人员未戴口罩时标记

## 架构

### 整体流程

```
人体检测线程 (已有) ─┬→ sleep_tracker → 睡岗检测线程 (新增)
                     ├→ mask_tracker  → 口罩判定逻辑 (复用 result_mask)
                     └→ 主循环渲染
```

### 睡岗组件

1. **模型加载**: 加载 `yolov8s-pose.pt` 作为姿态检测模型
2. **sleep_tracker**: 字典，跟踪每个人员的睡姿判定帧数
   ```python
   sleep_tracker[person_id] = {"sleep_count": 0, "sleeping": False}
   ```
3. **detect_sleep() 线程**: 读取 `result_human` 中的人体框，在裁剪区域上做姿态检测

### 睡姿判定规则

基于 YOLOv8-Pose 的 17 个关键点：
- 鼻子 (0) 的 y 坐标 > 左/右肩膀 (5,6) 的 y 坐标 → 头部低于肩膀
- 身体倾斜角 > 60° → 侧趴/趴着

满足上述条件之一即判定为该帧睡姿。

### 睡岗持续状态判定

- 检测到睡姿：`sleep_count += 1`
- 未检测到睡姿：`sleep_count = max(0, sleep_count - 2)`（允许偶尔几帧检测不到）
- `sleep_count >= 150`（约 5 秒 @30fps）：标记 `sleeping = True`
- `sleeping = True` 后，仅当 `sleep_count` 降为 0 时清除状态

### 未戴口罩判定逻辑

复用现有 `result_human`（人体框）和 `result_mask`（口罩框）：
- 对每个人体框，计算与所有口罩框的 IoU（交并比）
- IoU > 20% → 该人戴了口罩 ✅
- 所有口罩框的 IoU ≤ 20% → 该人未戴口罩 ❌

### 未戴口罩持续状态判定

- 未检测到口罩：`no_mask_count += 1`
- 检测到口罩：`no_mask_count = max(0, no_mask_count - 2)`
- `no_mask_count >= 30`（约 1 秒 @30fps）：标记未戴口罩
- 标记后，仅当计数降为 0 时清除状态

### 渲染规则

- 一个人只用一个框（复用人体检测的 bounding box）
- 🟢 绿色 = 正常（戴口罩、未睡岗），标注 `person 0.95`
- 🟡 黄色 = 异常（睡岗 或 未戴口罩），标注 `SLEEP 0.88` 或 `NO MASK 0.88`
- 不画第二个框，只在原人体框上改变颜色和文字

### 依赖

- `ultralytics`（已有，YOLOv8-Pose 使用同一库）
- 需要下载 `yolov8s-pose.pt` 权重文件（首次运行自动下载）
