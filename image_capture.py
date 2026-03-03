import cv2
import os
from insightface.app import FaceAnalysis

# ----------------------------
# CONFIG
# ----------------------------
person_name = "yakhsita"   # change this per person
dataset_path = "dataset"
url = "http://192.168.29.159:4747/video"

# ----------------------------
# Initialize InsightFace
# ----------------------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # 0 = CPU (Windows laptop)

# ----------------------------
# Create folder
# ----------------------------
person_folder = os.path.join(dataset_path, person_name)
os.makedirs(person_folder, exist_ok=True)

# ----------------------------
# Open DroidCam Stream
# ----------------------------
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Failed to open stream")
    exit()

count = 0

print("Press SPACE to capture face")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    display = frame.copy()

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Capture Faces", display)

    key = cv2.waitKey(1)

    if key == ord(' '):
        if len(faces) > 0:
            face = faces[0]
            x1, y1, x2, y2 = face.bbox.astype(int)

            face_crop = frame[y1:y2, x1:x2]

            img_path = os.path.join(person_folder, f"{count}.jpg")
            cv2.imwrite(img_path, face_crop)

            print(f"Saved {img_path}")
            count += 1
        else:
            print("No face detected. Try again.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
