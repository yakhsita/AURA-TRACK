import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

url = "http://192.168.162.243:4747/video"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Failed to open stream")
    exit()

print("Stream opened successfully")

frame_count = 0
deadband = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    frame = cv2.resize(frame, (640, 480))

    results = model(frame, imgsz=416, classes=[0])

    # Use YOLO's annotated frame (smooth box rendering)
    annotated = results[0].plot()

    frame_h, frame_w = annotated.shape[:2]
    cx_frame = frame_w // 2
    cy_frame = frame_h // 2

    # Draw green dot (camera center)
    cv2.circle(annotated, (cx_frame, cy_frame), 6, (0,255,0), -1)

    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Red dot = center of YOLO bounding box
        cx_box = (x1 + x2) // 2
        cy_box = (y1 + y2) // 2

        cv2.circle(annotated, (cx_box, cy_box), 6, (0,0,255), -1)

        # Line connecting dots
        cv2.line(annotated, (cx_frame, cy_frame), (cx_box, cy_box), (255,0,0), 2)

        # Horizontal error
        error_x = cx_box - cx_frame

        if abs(error_x) > deadband:
            if error_x < 0:
                print("Move Left")
            else:
                print("Move Right")
        else:
            print("Horizontally Aligned")

        # Check lock (green dot inside bounding box)
        if (x1 < cx_frame < x2) and (y1 < cy_frame < y2):
            print("Target Locked ✅")
        else:
            print("Aligning...")

    cv2.imshow("Aura-Track Vision", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
