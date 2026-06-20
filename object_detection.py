import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blurred = cv2.GaussianBlur(frame, (55, 55), 0)

    results = model.track(frame, persist=True, conf=0.4, tracker="bytetrack.yaml")
    result = results[0]

    output = blurred.copy()

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else -1

        output[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

        label = f"id:{track_id} {cls_name} {conf:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("CodeAlpha - Object Detection & Tracking", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
