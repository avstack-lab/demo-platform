#!/usr/bin/env python3

import argparse
import glob
import logging
import os
import time

import cv2
import numpy as np
import zmq
from cv2 import imread

from jumpstreet.context import SerializingContext
from jumpstreet.utils import BaseClass, TimeMonitor, config_as_namespace, init_some_end


img_exts = [".jpg", ".jpeg", ".png", ".tiff"]


class NearRealTimeImageLoader:
    """Loads images at nearly the correct rate

    It is expected that this will perform the necessary sleep
    process to enable near-correct-time sending
    """

    def __init__(self, image_paths, rate) -> None:
        self.image_paths = image_paths
        self.rate = rate
        self.interval = 1.0 / rate
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
        channel_order = "bgr"  # most likely loads as BGR since cv2
        self.counter += 1
        self.i_next_img = (self.i_next_img + 1) % len(self.image_paths)
        t_post = time.time()
        if self.t0 is None:
            self.t0 = t_post
        self.dt_last_load = t_post - t_pre_2
        self.next_target_send = self.t0 + self.counter * self.interval
        return data, channel_order


class SensorDataReplayer(BaseClass):
    """Replays sensor data from a folder"""

    NAME = "data-replayer"

    def __init__(
        self,
        context,
        backend,
        identifier,
        rate,
        send_dir,
        pattern=zmq.PUB,
        verbose=False,
        debug=False,
    ) -> None:
        super().__init__(self.NAME, identifier=identifier, verbose=verbose, debug=debug)
        self.pattern = pattern
        self.backend = init_some_end(
            self,
            context,
            "backend",
            pattern,
            backend.transport,
            backend.host,
            backend.port,
            BIND=backend.bind,
        )
        images = sorted(
            [
                img
                for ext in img_exts
                for img in glob.glob(os.path.join(send_dir, "*" + ext))
            ]
        )
        if len(images) == 0:
            raise RuntimeError(f"No images were found in {send_dir}!")
        self.rate = rate
        self.image_loader = NearRealTimeImageLoader(image_paths=images, rate=rate)
        self.time_monitor = TimeMonitor()
        if pattern == zmq.REQ:
            self._send_image_data(np.array([]), "READY")
            ack = self.backend.recv_multipart()
            assert ack[0] == b"OK"
            self.print("confirmed data broker ready", end="\n")

    def send(self):
        # -- load data
        data, channel_order = self.image_loader.load_next()
        a = 700  # this is bogus...fix later...f*mx
        b = 700  # this is bofus...fix later...f*my
        u = data.shape[1] / 2
        v = data.shape[0] / 2
        g = 0

        # -- send data
        ts = self.image_loader.counter / self.rate
        frame = self.image_loader.counter
        if self.debug:
            self.print(
                f"sending data, frame: {frame:4d}, timestamp: {ts:.4f}", end="\n"
            )

        msg = {
            "timestamp": ts,
            "frame": frame,
            "channel_order": channel_order,
            "identifier": self.identifier,
            "intrinsics": [a, b, g, u, v],
        }
        self._send_image_data(data, msg)
        self.time_monitor.trigger()

        # -- acknowledge
        if self.pattern == zmq.REQ:
            if self.debug:
                self.print("waiting for acknowledge...", end="")
            ack = self.backend.recv()
            assert ack == b"OK"
            if self.debug:
                print("done")

    def _send_image_data(self, array, msg):
        # -- image compression
        success, result = cv2.imencode(".jpg", array, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            raise RuntimeError("Error compressing image")
        compressed_frame = np.array(result)
        array = np.ascontiguousarray(compressed_frame)
        if array.flags["C_CONTIGUOUS"]:
            # if array is already contiguous in memory just send it
            self.backend.send_array(array, msg, copy=False)
        else:
            # else make it contiguous before sending
            array = np.ascontiguousarray(array)
            self.backend.send_array(array, msg, copy=False)


def main(config, sensor_id):
    """Runs sending on a single client"""
    context = SerializingContext(config.io_threads)
    replayer = SensorDataReplayer(
        context,
        backend=config.backend,
        identifier=sensor_id,
        rate=config.fps,
        send_dir=config.data_path,
        verbose=config.verbose,
        debug=config.debug,
    )
    try:
        while True:
            replayer.send()
    except Exception as e:
        logging.warning(e, exc_info=True)
    finally:
        replayer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Initialize sensor replayer client")
    parser.add_argument("--config", default="sensors/MOT15-replay.yml")
    parser.add_argument(
        "--sensor_id", default="camera_1", help="Identifier of the camera"
    )
    args = parser.parse_args()
    config = config_as_namespace(args.config)
    sensor_id = args.sensor_id
    main(config, sensor_id)
