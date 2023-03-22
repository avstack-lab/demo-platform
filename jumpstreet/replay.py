#!/usr/bin/env python3

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
# from jumpstreet.utils import BaseClass, init_some_end
from context import SerializingContext
from utils import BaseClass, init_some_end


img_exts = [".jpg", ".jpeg", ".png", ".tiff"]


class NearRealTimeImageLoader():
    """Loads images at nearly the correct rate
    
    It is expected that this will perform the necessary sleep
    process to enable near-correct-time sending
    """
    def __init__(self, image_paths, rate) -> None:
        self.image_paths = image_paths
        self.rate = rate
        self.interval = 1./rate
        self.i_next_img = 0
        self.counter = 0
        self.last_load_time = 0
        self.t0 = None
        self.dt_last_load = 0
        self.next_target_send = None
        
    def load_next(self):
        t_pre_1 = time.time()
        if self.next_target_send is not None:
            dt_wait = self.next_target_send - t_pre_1 - self.dt_last_load
            if dt_wait > 0:
                time.sleep(dt_wait)
        t_pre_2 = time.time()
        data = imread(self.image_paths[self.i_next_img])
        self.counter += 1
        self.i_next_img = (self.i_next_img + 1) % len(self.image_paths)
        t_post = time.time()
        if self.t0 is None:
            self.t0 = t_post
        self.dt_last_load = t_post - t_pre_2
        self.next_target_send = self.t0 + self.counter * self.interval
        return data


class SensorDataReplayer(BaseClass):
    """Replays sensor data from a folder"""

    NAME = "data-replayer"

    def __init__(self, context, HOST, PORT, identifier, send_dir, pattern=zmq.PUB, rate=10, verbose=False) -> None:
        super().__init__(self.NAME, identifier=identifier, verbose=verbose)
        self.pattern = pattern
        self.backend = init_some_end(
            cls=self,
            context=context,
            end_type="backend",
            pattern=pattern,
            HOST=HOST,
            PORT=PORT,
            BIND=False,
        )
        images = sorted(
            [
                img
                for ext in img_exts
                for img in glob.glob(os.path.join(send_dir, "*" + ext))
            ]
        )
        if len(images) == 0:
            raise RuntimeError(f'No images were found in {send_dir}!')
        self.rate = rate
        self.image_loader = NearRealTimeImageLoader(image_paths=images, rate=rate)
        if pattern == zmq.REQ:
            self._send_image_data(np.array([]), "READY")
            ack = self.backend.recv_multipart()
            assert ack[0] == b"OK"
            self.print("confirmed data broker ready", end="\n")

    def send(self):
        # -- load data
        data = self.image_loader.load_next()
        a = 700  # this is bogus...fix later...f*mx
        b = 700  # this is bofus...fix later...f*my
        u = data.shape[1]/2
        v = data.shape[0]/2
        g = 0
        
        # -- send data
        if self.verbose:
            self.print("sending data...", end="")
        msg = {'timestamp':self.image_loader.counter/self.rate,
               'frame':self.image_loader.counter,
               'identifier':self.identifier,
               'intrinsics':[a, b, g, u, v]}
        self._send_image_data(data, msg)
        if self.verbose:
            print("done")

        # -- acknowledge
        if self.pattern == zmq.REQ:
            if self.verbose:
                self.print("waiting for acknowledge...", end="")
            ack = self.backend.recv()
            assert ack == b"OK"
            if self.verbose:
                print("done")

    def _send_image_data(self, array, msg):
        if array.flags["C_CONTIGUOUS"]:
            # if array is already contiguous in memory just send it
            self.backend.send_array(array, msg, copy=False)
        else:
            # else make it contiguous before sending
            array = np.ascontiguousarray(array)
            self.backend.send_array(array, msg, copy=False)


def start_client(task, *args):
    """Starting a client using multiproc"""
    process = multiprocessing.Process(target=task, args=args)
    process.daemon = True
    process.start()


def main_single(HOST, PORT, identifier, send_rate, send_dir, verbose):
    """Runs sending on a single client"""
    context = SerializingContext()
    replayer = SensorDataReplayer(
        context, HOST=HOST, PORT=PORT, identifier=identifier, send_dir=send_dir, rate=send_rate, verbose=verbose,
    )
    try:
        while True:
            replayer.send()
    except Exception as e:
        logging.warning(e, exc_info=True)
    finally:
        replayer.close()


def main(args):
    """Run sensor replayer clients"""
    for i in range(args.nclients):
        start_client(
            main_single, args.host, args.port, i, args.send_rate, args.send_dir, args.verbose,
        )
    while True:
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Initialize sensor replayer client")
    parser.add_argument(
        "-n", "--nclients", type=int, default=1, help="Number of clients"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", type=str, help="Hostname to connect to"
    )
    parser.add_argument(
        "--port", default=5550, type=int, help="Port to connect to server/broker"
    )
    parser.add_argument(
        "--send_rate", default=10, type=int, help="Replay rate for sensor data"
    )
    parser.add_argument(
        "--send_dir",
        type=str,
        # default="./data/ADL-Rundle-6/img1",
        default="./data/data/tracking/MOT15/train/ADL-Rundle-6/img1",
        help="Directory for data replay",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)
