#!/usr/bin/env python3

import argparse
import glob
import os
from time import sleep
from cv2 import imread
import zmq
import jumpstreet.messages as jmess
import imagezmq


def printhere(msg, end='\n'):
    print('::replayer::' + msg, end=end)
    

class Replayer:
    """Replays images from a buffer for testing"""

    def __init__(self, image_dir, PM_PORT=5555, exts=("*.png", "*.jpg", "*.jpeg")) -> None:
        self.name = 'Image Replayer'
        self.image_dir = image_dir
        self.file_buffer = []
        for ext in exts:
            self.file_buffer.extend(glob.glob(os.path.join(image_dir, ext)))
        self.file_buffer = sorted(self.file_buffer)
        if len(self.file_buffer) == 0:
            raise RuntimeError(f'::replayer::could not find any images at {image_dir}')
        else:
            printhere(f'found {len(self.file_buffer)} images!')
        self.i_last = -1
        self.n_sent = 0

        # TCP connection to process manager
        printhere('establishing TCP connection to process manager...', end='')
        self.pm_zmq_context = zmq.Context()
        self.pm_zmq_socket = self.pm_zmq_context.socket(zmq.REQ)
        flag = self.pm_zmq_socket.connect(f"tcp://localhost:{PM_PORT}")
        print('done')

        # Get broadcast topic for images over TCP
        self.pm_zmq_socket.send(jmess.SensorStartup(self.name).encode())
        json_msg = self.pm_zmq_socket.recv()
        msg = jmess.decode(json_msg)
        if isinstance(msg, jmess.SensorStartupReply):
            assert msg.sensor_name == self.name
            # Establish broadcast topic on UDP
            DATA_PORT = msg.DATA_PORT
            printhere(f'establishing UDP topic for {self.name} on port {DATA_PORT}...', end='')
            self.data_zmq_sender = imagezmq.ImageSender(
                connect_to=f'tcp://127.0.0.1:{DATA_PORT}', REQ_REP=False)
            print('done')
        else:
            raise jmess.UnknownMessageError(msg)

    def send(self) -> bool:
        """Send next image from buffer"""
        next_img = self.i_last+1
        printhere(f'sending image {next_img}')
        img = imread(self.file_buffer[next_img])
        self.data_zmq_sender.send_image(self.name, img)
        self.i_last = next_img
        self.n_sent += 1


def main(args):
    dt = 1./args.rate
    rep = Replayer(args.image_dir, PM_PORT=args.pm_port)
    while True:
        exhausted = rep.send()
        sleep(dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay image data from buffer")
    parser.add_argument("image_dir", type=str, help="Where to find the image data")
    parser.add_argument(
        "--rate", required=True, type=float, help="Rate at which to send the data (fps)"
    )
    parser.add_argument(
        "--circular",
        action="store_true",
        help="Whether to treat the buffer as circular. Otherwise, resets.",
    )
    parser.add_argument(
        "--pm_port",
        default=5555,
        type=int,
        help="Port number for process manager TCP connection"
    )

    args = parser.parse_args()
    main(args)
