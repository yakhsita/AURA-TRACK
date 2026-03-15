import cv2
import subprocess
from ultralytics import YOLO

WINDOWS_IP = "10.122.207.71"
PORT = 5000
width = 640
height = 480
fps = 30

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

if not cap.isOpened():
    print("Camera failed to open")
    exit()

ffmpeg_cmd = [
    "ffmpeg",
    "-f", "rawvideo",
    "-pixel_format", "bgr24",
    "-video_size", f"{width}x{height}",
    "-framerate", str(fps),
    "-i", "pipe:0",
    "-f", "rtp",
    "-vcodec", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-b:v", "4000k",
"-maxrate", "800k",
    "-bufsize", "400k",    # small buffer = low latency
    "-g", "15",            # keyframe every 15 frames, faster recovery
    "-pix_fmt", "yuv420p",
    f"rtp://{WINDOWS_IP}:{PORT}"
]

proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

print("Streaming started...")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        results = model(frame, verbose=False)
        annotated = results[0].plot()
        proc.stdin.write(annotated.tobytes())
except KeyboardInterrupt:
    print("Stopping...")
finally:
    cap.release()
    proc.stdin.close()
    proc.wait()
