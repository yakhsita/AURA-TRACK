import cv2
from ultralytics import YOLO

# ----------------------------
# Load YOLO model (CPU)
# ----------------------------
model = YOLO("yolov8n.pt")  # CPU inference

# ----------------------------
# Video stream
# ----------------------------
url = "http://192.168.162.243:4747/video"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Failed to open stream")
    exit()

print("Stream opened successfully")

# ----------------------------
# Tracking variables
# ----------------------------
tracker = None
tracking = False
frame_count = 0
deadband = 50
update_every = 5  # YOLO correction frequency
bounding_box = None  # last known box

# ----------------------------
# Main loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Resize for faster processing
    frame = cv2.resize(frame, (480, 360))
    key = cv2.waitKey(1) & 0xFF

    # ----------------------------
    # NOT TRACKING → detection mode
    # ----------------------------
    if not tracking:
        results = model(frame, imgsz=416, classes=[0])
        annotated = results[0].plot()

        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Press S to lock target
            if key == ord('s'):
                tracker = cv2.legacy.TrackerMOSSE_create()
                tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                tracking = True
                bounding_box = (x1, y1, x2 - x1, y2 - y1)
                print("Target Locked 🔒")

    # ----------------------------
    # TRACKING MODE
    # ----------------------------
    else:
        success, box = tracker.update(frame)
        annotated = frame.copy()

        if success:
            x, y, w, h = map(int, box)
            bounding_box = (x, y, w, h)

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Frame center (green dot)
            frame_h, frame_w = annotated.shape[:2]
            cx_frame = frame_w // 2
            cy_frame = frame_h // 2
            cv2.circle(annotated, (cx_frame, cy_frame), 6, (0, 255, 0), -1)

            # Person center (red dot)
            cx_box = x + w // 2
            cy_box = y + h // 2
            cv2.circle(annotated, (cx_box, cy_box), 6, (0, 0, 255), -1)

            # Line connecting dots
            cv2.line(annotated, (cx_frame, cy_frame), (cx_box, cy_box), (255, 0, 0), 2)

            # Horizontal alignment
            error_x = cx_box - cx_frame
            if abs(error_x) > deadband:
                if error_x < 0:
                    print("Move Left")
                else:
                    print("Move Right")
            else:
                print("Horizontally Aligned")

            # Vertical alignment (optional)
            error_y = cy_box - cy_frame
            if abs(error_y) > deadband:
                if error_y < 0:
                    print("Move Up")
                else:
                    print("Move Down")

        else:
            print("Tracking Lost ❌")
            tracking = False
            tracker = None

        # ----------------------------
        # Safe YOLO correction every N frames
        # ----------------------------
        if tracking and frame_count % update_every == 0:
            results = model(frame, imgsz=416, classes=[0])
            if len(results[0].boxes) > 0:
                # Find closest box to current bounding box
                min_dist = float('inf')
                closest_box = None
                for b in results[0].boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    prev_cx = bounding_box[0] + bounding_box[2] // 2
                    prev_cy = bounding_box[1] + bounding_box[3] // 2
                    dist = (cx - prev_cx) ** 2 + (cy - prev_cy) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        closest_box = (x1, y1, x2, y2)
                # Update tracker coordinates only without recreating
                if closest_box is not None:
                    x1, y1, x2, y2 = closest_box
                    bounding_box = (x1, y1, x2 - x1, y2 - y1)
                    tracker.init(frame, bounding_box)  # safe coordinate update
                    print("Tracker coordinates updated with YOLO 🔧")

    # ----------------------------
    # Reset tracking manually
    # ----------------------------
    if key == ord('r'):
        tracking = False
        tracker = None
        print("Tracking Reset")

    # Display frame
    cv2.imshow("Aura-Track Vision 🚁", annotated)

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
