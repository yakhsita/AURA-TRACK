import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# Get absolute path to current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(BASE_DIR, "dataset")

app = FaceAnalysis(name='buffalo_sc')
app.prepare(ctx_id=-1)

known_embeddings = []
known_names = []

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    for image_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, image_name)

        print("Processing:", img_path)

        img = cv2.imread(img_path)

        if img is None:
            print("Failed to load:", img_path)
            continue

        faces = app.get(img)

        if len(faces) > 0:
            known_embeddings.append(faces[0].embedding)
            known_names.append(person_name)
            print("Added:", image_name)
        else:
            print("No face found in:", image_name)

data = {
    "embeddings": np.array(known_embeddings),
    "names": known_names
}

with open(os.path.join(BASE_DIR, "encodings.pickle"), "wb") as f:
    pickle.dump(data, f)

print("Encodings saved.")
