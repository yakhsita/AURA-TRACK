import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

# ----------------------------
# Load Face Model
# ----------------------------
app = FaceAnalysis(name='buffalo_sc')
app.prepare(ctx_id=-1)

data = pickle.load(open("encodings.pickle", "rb"))
known_embeddings = np.array(data["embeddings"])
known_embeddings = known_embeddings / np.linalg.norm(known_embeddings, axis=1, keepdims=True)
known_names = data["names"]

criminal = "yakhsita"
deadband = 50
locked = False
locked_box = None

def recognize_face(embedding):
    embedding = embedding / np.linalg.norm(embedding)
    similarities = np.dot(known_embeddings, embedding)
    best_match = np.argmax(similarities)
    score = similarities[best_match]

    if score > 0.42:
        return known_names[best_match]
    else:
        return "Unknown"

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1+x2)//2, (y1+y2)//2)

url = "http://192.168.29.159:4747/video"
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (480, 360))
    faces = app.get(frame)

    if not locked:
        # SEARCH MODE
        for face in faces:
            bbox = face.bbox.astype(int)
            name = recognize_face(face.embedding)

            if name == criminal:
                locked = True
                locked_box = bbox
                print("TARGET LOCKED 🔒")
                break

    else:
        # TRACK MODE
        if len(faces) == 0:
            print("Lost Target ❌")
            locked = False
            locked_box = None
        else:
            min_dist = float("inf")
            best_box = None

            old_center = box_center(locked_box)

            for face in faces:
                bbox = face.bbox.astype(int)
                new_center = box_center(bbox)

                dist = np.linalg.norm(
                    np.array(old_center) - np.array(new_center)
                )

                if dist < min_dist:
                    min_dist = dist
                    best_box = bbox

            if min_dist < 120:
                locked_box = best_box
            else:
                print("Lost Target ❌")
                locked = False
                locked_box = None

    # DRAW
    if locked and locked_box is not None:
        x1, y1, x2, y2 = locked_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # Frame center (green)
        frame_h, frame_w = frame.shape[:2]
        cx_frame = frame_w // 2
        cy_frame = frame_h // 2
        cv2.circle(frame, (cx_frame, cy_frame), 6, (0,255,0), -1)

        # Target center (red)
        cx_box = (x1+x2)//2
        cy_box = (y1+y2)//2
        cv2.circle(frame, (cx_box, cy_box), 6, (0,0,255), -1)

        # Connect line
        cv2.line(frame, (cx_frame, cy_frame), (cx_box, cy_box), (255,0,0), 2)

        # Horizontal logic
        error_x = cx_box - cx_frame
        if abs(error_x) > deadband:
            if error_x < 0:
                print("Move Left")
            else:
                print("Move Right")
        else:
            print("Horizontally Aligned")

        # Vertical logic
        error_y = cy_box - cy_frame
        if abs(error_y) > deadband:
            if error_y < 0:
                print("Move Up")
            else:
                print("Move Down")

    cv2.imshow("Aura-Track Vision 🚁", frame)

    if cv2.waitKey(1) & 0xFF == ord('r'):
        locked = False
        locked_box = None
        print("Tracking Reset")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
