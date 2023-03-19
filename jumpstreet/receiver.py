#!/usr/bin/env python3

"""
Author: Nate Zelter
Date: February 2023

"""


import argparse
import glob
import logging
import multiprocessing
import os
import time

import numpy as np
import zmq
from cv2 import imread

# from jumpstreet.context import SerializingContext
from utils import BaseClass, init_some_end


def main(args):
    pass
    print("Hello world!")
    print(args)
    return 0

class SensorDataReceiver(BaseClass):
    """Replays sensor data from a folder"""

    NAME = "data-receiver"

    def __init__(self, context, HOST, PORT, identifier, send_dir, pattern=zmq.SUB, rate=20) -> None:
        super().__init__(self.NAME, identifier)
        self.pattern = pattern
        self.frontend = init_some_end(
            cls=self,
            context=context,
            end_type="frontend", # was "backend" for SensorDataReplayer in replay.py
            pattern=pattern,
            HOST=HOST,
            PORT=PORT,
            BIND=False,
        )
        
        self.rate = rate
        # if pattern == zmq.REQ:
        #     self._send_image_data(np.array([]), "READY")
        #     ack = self.backend.recv_multipart()
        #     assert ack[0] == b"OK"
        #     self.print("confirmed data broker ready", end="\n")

    def send(self):
        # -- load data
        data = self.image_loader.load_next()

        # -- send data
        self.print("sending data...", end="")
        self._send_image_data(data, f"TIME_{self.image_loader.i_next_img/self.rate:.2f}_CAM_{self.identifier:02d}")
        print("done")

        # -- acknowledge
        if self.pattern == zmq.REQ:
            self.print("waiting for acknowledge...", end="")
            ack = self.backend.recv()
            assert ack == b"OK"
            print("done")

    def _send_image_data(self, array, msg):
        if array.flags["C_CONTIGUOUS"]:
            # if array is already contiguous in memory just send it
            self.backend.send_array(array, msg, copy=False)
        else:
            # else make it contiguous before sending
            array = np.ascontiguousarray(array)
            self.backend.send_array(array, msg, copy=False)






if __name__ == "__main__":
    parser = argparse.ArgumentParser("Initialize sensor data receiver client")
    parser.add_argument(
        "-n", "--nclients", type=int, default=1, help="Number of clients"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", type=str, help="Hostname to connect to"
    )
    parser.add_argument(
        "--port", default=6550, type=int, help="Port to connect to server/broker"
    )
    # parser.add_argument(
    #     "--send_rate", default=10, type=int, help="Replay rate for sensor data"
    # )
    # parser.add_argument(
    #     "--send_dir",
    #     type=str,
    #     default="./data/ADL-Rundle-6/img1",
    #     help="Directory for data replay",
    # )

    args = parser.parse_args()
    main(args)