import cv2
import zmq
import numpy as np

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:6558")
socket.setsockopt_string(zmq.SUBSCRIBE, '')

while True:
    try:
        # Receive message and image
        message = socket.recv()
        img = socket.recv(copy=False)
        npimg = np.frombuffer(img, dtype=np.uint8)
        decoded_frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Display image
        cv2.imshow('Camera stream', npimg)
        # cv2.imshow('Camera Stream', decoded_frame)

        # Wait for key event and break on 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()
