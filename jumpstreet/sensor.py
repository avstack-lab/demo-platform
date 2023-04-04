#!/usr/bin/env python3

"""
Author: Nate Zelter
Date: March 2023

"""

import argparse
import time

import cv2
import numpy as np
import PySpin
import zmq
from context import SerializingContext
from utils import BaseClass, init_some_end, send_array_pubsub, send_jpg_pubsub
import sys

STOP_KEY = "q"
DEFAULT_BACKEND_PORT = 6551
ACCEPTABLE_SENSOR_TYPES = ["camera-flir-bfs", "camera-rpi"]


def flir_capture(handle, image_dimensions):
    handle.BeginAcquisition()

    for i in range(50):
        ptr = handle.GetNextImage()
        arr = np.frombuffer(ptr.GetData(), dtype=np.uint8).reshape(image_dimensions)
        img = cv2.cvtColor(arr, cv2.COLOR_BayerBG2BGR)  # np.ndarray
        if not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img)
        msg = "sample"
        image_show = cv2.resize(img, None, fx=0.25, fy=0.25)
        cv2.imshow(f"Press {STOP_KEY} to quit", image_show)
        key = cv2.waitKey(30)
        if key == ord(STOP_KEY):
            print("Received STOP_KEY signal")
            ptr.Release()
            handle.EndAcquisition()
            break
        ptr.Release()
    handle.EndAcquisition()
    handle.DeInit()


class Sensor(BaseClass):
    """
    Sensor ZMQ Node, device agnostic
    Pattern: Data aquisition --> context.socket(zmq.PUB)
    """

    NAME = "sensor"

    def __init__(
        self,
        context,
        identifier,
        type,
        configs,
        host,
        backend,
        backend_other,
        resize_factor,
        verbose=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(self.NAME, identifier)

        if type in ACCEPTABLE_SENSOR_TYPES:
            self.type = type
        else:
            raise NotImplementedError(f"Unacceptable sensor type: {type}")

        self.configs = configs

        self.backend = init_some_end(
            self, context, "backend", zmq.PUB, host, backend, BIND=False
        )

        if backend_other is not None:
            self.backend_other = init_some_end(
                self, context, "backend_other", zmq.PUB, host, backend_other, BIND=False
            )

        self.resize_factor = resize_factor
        self.verbose = verbose
        self.handle = None
        self.streaming = False

    def initialize(self):
        if self.type == "camera-flir-bfs":

            ## Extract sensor configs for flir bfs
            cam_name = self.configs.get("name", "NA")
            cam_serial = self.configs.get("serial", "NA")
            cam_ip = self.configs.get("ip", "NA")
            cam_width_px = int(self.configs.get("width_px", 0))
            cam_height_px = int(self.configs.get("height_px", 0))
            cam_fps = int(self.configs.get("fps", 0))
            cam_frame_size_bytes = int(self.configs.get("frame_size_bytes", 0))

            ## Connect to camera
            system = PySpin.System.GetInstance()
            cam_list = system.GetCameras()
            self.handle = cam_list.GetBySerial(cam_serial)
            try:
                self.handle.Init()
                print(f"Successfully connected to {cam_name} via serial number")
            except:
                raise RuntimeError(f"Unable to connect to {cam_name} via serial number")

            ## Set the camera properties here
            self.handle.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            self.handle.Width.SetValue(cam_width_px)
            self.handle.Height.SetValue(cam_height_px)
            self.handle.AcquisitionFrameRateEnable.SetValue(
                True
            )  # enable changes to FPS
            self.handle.AcquisitionFrameRate.SetValue(
                cam_fps
            )  # max is 24fps for FLIR BFS

            self.image_dimensions = (cam_height_px, cam_width_px)

            #! Method should end here
            #### --------------------------------------------------------------
        
            # TODO Fill this in later....
            a = 700  # this is bogus...fix later...f*mx
            b = 700  # this is bofus...fix later...f*my
            u = cam_width_px / 2
            v = cam_height_px / 2
            g = 0
            msg = {
                "timestamp": 0.0,
                "frame": 0,
                "identifier": self.identifier,
                "intrinsics": [a, b, g, u, v],
                "channel_order": "rgb",
            }

            
            self.handle.BeginAcquisition()
            t0 = 0
            frame_counter = 0
            while True:

                ptr = self.handle.GetNextImage()
                if ptr.IsIncomplete():
                    continue # discard image

                ts_raw = float(ptr.GetTimeStamp())
                timestamp = ts_raw * 1e-9 # ms
                if frame_counter == 0:
                    t0 = timestamp
                msg["timestamp"] = round(timestamp - t0, 9)
                # print(ts_raw)
                msg["frame"] = frame_counter

                # -- Version 1: successfully gets colored image as ndarray
                arr = np.frombuffer(ptr.GetData(), dtype=np.uint8).reshape(self.image_dimensions)
                img = cv2.cvtColor(arr, cv2.COLOR_BayerBG2BGR)  # np.ndarray with d = (h, w, 3)
 
                # -- resize image before compression
                new_h = int(img.shape[0] / self.resize_factor)
                new_w = int(img.shape[1] / self.resize_factor)
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                img = img_resized

                # -- image compression
                success, result = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not success:
                    raise RuntimeError("Error compressing image")
                compressed_frame = np.array(result)
                img = np.ascontiguousarray(compressed_frame)
             


                self.backend.send_array(img, msg, False)
                if self.verbose:
                    self.print(f"sending data, frame: {frame_counter:4d}, timestamp: {timestamp:.4f}", end="\n")
                ptr.Release()
                frame_counter += 1

        elif self.type == "camera-rpi":
            pass
        else:
            pass

    def start_capture(self):
        print("entered start_capture() ")

        if self.type == "camera-flir-bfs":
            a = 700  # this is bogus...fix later...f*mx
            b = 700  # this is bofus...fix later...f*my
            u = self.image_dimensions[1] / 2
            v = self.image_dimensions[0] / 2
            g = 0
            msg = {
                "timestamp": 0.0,
                "frame": 0,
                "identifier": self.identifier,
                "intrinsics": [a, b, g, u, v],
                "channel_order": "rgb",
                "compression": "jpeg"
            }

            #! Method should end here
            self.handle.BeginAcquisition()
            t0 = 0
            frame_counter = 0
            while True:

                ptr = self.handle.GetNextImage()

                timestamp = float(ptr.GetTimeStamp()) * 1e-9 # ms
                if frame_counter == 0:
                    t0 = timestamp
                msg["timestamp"] = round(timestamp - t0, 9)
                print(timestamp)
                msg["frame"] = frame_counter

                arr = np.frombuffer(ptr.GetData(), dtype=np.uint8).reshape(
                    self.image_dimensions
                )
                img = cv2.cvtColor(arr, cv2.COLOR_BayerBG2BGR)  # np.ndarray
                ret, jpeg_buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if not ret:
                        raise RuntimeError("Error compressing image")
                compressed_frame = np.array(jpeg_buffer)
                img = np.ascontiguousarray(compressed_frame)

                # if not img.flags["C_CONTIGUOUS"]:
                #     img = np.ascontiguousarray(img)

                self.backend.send_array(img, msg)
                if self.verbose:
                    self.print(f"sending data, frame: {frame_counter:4d}, timestamp: {timestamp:.4f}", end="\n")

                frame_counter += 1

        elif self.type == "camera-rpi":
            pass
        else:
            pass

    def stop_capture(self):
        pass

    def reconfigure():
        pass


def main(args, configs):

    ### Instantiate Sensor and configure device ###
    context = SerializingContext()
    sensor = Sensor(
        context,
        configs["name"],
        args.type,
        configs,
        args.host,
        args.backend,
        args.backend_other,
        args.resize_factor,
        verbose=args.verbose,
    )

    sensor.initialize()
    print("Sensor successfully initialized in sensor.py")

    sensor.start_capture()
    # print("foo")


if __name__ == "__main__":

    ## -- load configuration from json (this will later be done by controller)
    configs = {
        "camera_1": {
            "name": "camera_1",
            "type": "FLIR-BFS-50S50C",
            "serial": "22395929",
            "ip": "192.168.1.1",
            "width_px": "2448",
            "height_px": "2048",
            "fps": "20",
            "frame_size_bytes": "307200",
        },
        "camera_2": {
            "name": "camera_2",
            "type": "FLIR-BFS-50S50C",
            "serial": "22395953",
            "ip": "192.168.1.2",
            "width_px": "2448",
            "height_px": "2048",
            "fps": "10",
            "frame_size_bytes": "307200",
        },
        "camera_3": {
            "name": "camera_3",
            "type": "Raspberry-Pi",
            "serial": "22395953",
            "ip": "192.168.1.2",
            "width_px": "2448",
            "height_px": "2048",
            "fps": "10",
            "frame_size_bytes": "307200",
        },
    }

    parser = argparse.ArgumentParser("Initialize a Sensor")
    parser.add_argument(
        "--type",
        choices=ACCEPTABLE_SENSOR_TYPES,
        type=str,
        help="Selection of sensor type",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host")
    parser.add_argument(
        "--backend",
        type=int,
        default=DEFAULT_BACKEND_PORT,
        help="Backend port number (PUB)",
    )
    parser.add_argument(
        "--backend_other",
        type=int,
        help="Extra backend port (used only in select classes)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resize_factor", 
                        choices = [1, 2, 4, 8],
                        type=int,
                        help="Resize image by a factor of 1/x")

    args = parser.parse_args()

    main(args, configs["camera_1"])
