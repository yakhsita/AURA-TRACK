import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Camera stream
url = "http://192.168.162.243:4747/video"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Failed to open stream")
    exit()

print("Stream opened successfully")

# Tracking variables
tracker = None
tracking = False
frame_count = 0
deadband = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames for smoother performance
    if frame_count % 2 != 0:
        continue

    frame = cv2.resize(frame, (640, 480))

    key = cv2.waitKey(1) & 0xFF

    # ----------------------------
    # NOT TRACKING → DETECT MODE
    # ----------------------------
    if not tracking:

        results = model(frame, imgsz=416, classes=[0])
        annotated = results[0].plot()

        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Press S to lock first detected person
            if key == ord('s'):
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                tracking = True
                print("Target Locked 🔒")

    # ----------------------------
    # TRACKING MODE
    # ----------------------------
    else:
        success, box = tracker.update(frame)
        annotated = frame.copy()

        if success:
            x, y, w, h = map(int, box)

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Frame center (Green dot)
            frame_h, frame_w = annotated.shape[:2]
            cx_frame = frame_w // 2
            cy_frame = frame_h // 2
            cv2.circle(annotated, (cx_frame, cy_frame), 6, (0, 255, 0), -1)

            # Person center (Red dot)
            cx_box = x + w // 2
            cy_box = y + h // 2
            cv2.circle(annotated, (cx_box, cy_box), 6, (0, 0, 255), -1)

            # Line between dots
            cv2.line(annotated, (cx_frame, cy_frame), (cx_box, cy_box), (255, 0, 0), 2)

            # Alignment logic
            error_x = cx_box - cx_frame

            if abs(error_x) > deadband:
                if error_x < 0:
                    print("Move Left")
                else:
                    print("Move Right")
            else:
                print("Horizontally Aligned")

        else:
            print("Tracking Lost ❌")
            tracking = False

        # Press R to reset tracking
        if key == ord('r'):
            tracking = False
            tracker = None
            print("Tracking Reset")

    cv2.imshow("Aura-Track Vision 🚁", annotated)

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
