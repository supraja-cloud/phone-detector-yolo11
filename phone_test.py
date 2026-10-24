"""
Real-Time Phone Detection using YOLOv11 & OpenCV
Author: supraja-cloud
License: MIT License
Repository: phone-detector-yolo11
"""

import cv2, time, threading
from ultralytics import YOLO

MODEL = "yolo11s.pt"   # n = fastest, s = balanced, m = most accurate (slow on CPU)
CONF = 0.20
IMGSZ = 640            # 480 = faster, 800+ = better for small phones but slower
CONSECUTIVE = 2
PHONE_LIKE = {67, 65}  # cell phone, remote

import torch
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("Running on:", "GPU" if DEVICE == 0 else "CPU")

model = YOLO(MODEL)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

state = {"frame": None, "boxes": [], "hits": 0, "infer_fps": 0, "run": True}
lock = threading.Lock()

def infer_loop():
    last = time.time()
    while state["run"]:
        with lock:
            frame = None if state["frame"] is None else state["frame"].copy()
        if frame is None:
            time.sleep(0.01)
            continue
        res = model.predict(frame, conf=CONF, imgsz=IMGSZ, device=DEVICE,
                            half=(DEVICE == 0), verbose=False)[0]
        boxes = []
        for b in res.boxes:
            if int(b.cls[0]) in PHONE_LIKE:
                boxes.append((*map(int, b.xyxy[0]), float(b.conf[0])))
        now = time.time()
        with lock:
            state["boxes"] = boxes
            state["hits"] = state["hits"] + 1 if boxes else 0
            state["infer_fps"] = 1 / (now - last)
        last = now

threading.Thread(target=infer_loop, daemon=True).start()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    with lock:
        state["frame"] = frame
        boxes, hits, ifps = list(state["boxes"]), state["hits"], state["infer_fps"]

    for x1, y1, x2, y2, c in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"PHONE {c:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Detect FPS {ifps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if hits >= CONSECUTIVE:
        cv2.putText(frame, "PHONE DETECTED!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Phone Detection Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

state["run"] = False
cap.release()
cv2.destroyAllWindows()