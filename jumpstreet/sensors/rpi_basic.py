import time

import numpy as np
import picamera


# Define Raspberry Pi camera IP address
camera_ip = "192.168.1.2"

camera = picamera.PiCamera(camera_ip)
camera.resolution = (640, 480)
camera.framerate = 30
camera.start_preview()

# Wait for camera to warm up
time.sleep(2)

# Define a function to process each frame
def process_frame(frame):
    arr = np.frombuffer(frame.getvalue(), dtype=np.uint8)
    img = arr.reshape((480, 640, 3))
    processed_img = img[::-1, :, :]
    return processed_img


# Capture a continuous stream of frames and process each one
for frame in camera.capture_continuous(
    np.zeros((480, 640, 3), dtype=np.uint8), format="rgb", use_video_port=True
):
    print(len(frame))  # ndarray, d=(h, w, 3)
