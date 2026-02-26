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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip every 2nd frame (reduces lag)
    if frame_count % 2 != 0:
        continue

    # Resize for faster processing
    frame = cv2.resize(frame, (640, 480))

    results = model(frame, imgsz=416, classes=[0])
    if len(results[0].boxes) > 0:
    	box = results[0].boxes[0]  # First detected person

    	x1, y1, x2, y2 = box.xyxy[0]
    	person_center_x = int((x1 + x2) / 2)

    	frame_center_x = frame.shape[1] // 2

    	if person_center_x < frame_center_x - 50:
        	print("Move Left")
    	elif person_center_x > frame_center_x + 50:
        	print("Move Right")
    	else:
        	print("Stay Centered")

    annotated = results[0].plot()

    cv2.imshow("YOLO Live Demo", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
