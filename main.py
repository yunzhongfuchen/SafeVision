"""AI 视频监控 - 终端入口（保留兼容）"""
import cv2
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

if engine.cap is not None:
    engine.cap.release()
cv2.destroyAllWindows()
