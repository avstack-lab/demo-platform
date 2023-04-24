#!/usr/bin/env python3

"""
Author: Nate Zelter
Date: March 2023

"""

import argparse
import sys
import time

import cv2
import numpy as np
import PySpin
import rad
import zmq
from avstack.geometry.transformations import matrix_cartesian_to_spherical
from context import SerializingContext
from utils import BaseClass, init_some_end, send_array_pubsub, send_jpg_pubsub


STOP_KEY = "q"
DEFAULT_BACKEND_PORT = 6551
ACCEPTABLE_SENSOR_TYPES = ["camera-flir-bfs", "camera-rpi", "ti-radar"]


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

    NAME = "generic-sensor"

    def __init__(
        self,
        context,
        BACKEND_HOST,
        BACKEND_PORT,
        sensor_type,
        configs,
        identifier,
        verbose=False,
        debug=False,
    ) -> None:
        super().__init__(self.NAME, identifier, verbose=verbose, debug=debug)

        if sensor_type in ACCEPTABLE_SENSOR_TYPES:
            self.sensor_type = sensor_type
        else:
            raise NotImplementedError(f"Unacceptable sensor type: {sensor_type}")

        self.configuration = configs
        self.backend = init_some_end(
            self, context, "backend", zmq.PUB, BACKEND_HOST, BACKEND_PORT, BIND=False
        )

    def initialize(self):
        raise NotImplementedError

    def start_capture(self):
        raise NotImplementedError


class Radar(Sensor):
    NAME = "radar-sensor"

    def __init__(
        self,
        context,
        BACKEND_HOST,
        BACKEND_PORT,
        sensor_type,
        configs,
        identifier,
        verbose=False,
        debug=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            context,
            BACKEND_HOST,
            BACKEND_PORT,
            sensor_type,
            configs,
            identifier,
            verbose,
            debug,
        )
        self.radar = None
        self.frame = 0

    def initialize(self):
        self.radar = rad.Radar(
            config_file_name=self.configuration["config_file_name"],
            translate_from_JSON=False,
            enable_serial=True,
            CLI_port=self.configuration["CLI_port"],
            Data_port=self.configuration["Data_port"],
            enable_plotting=False,
            jupyter=False,
            data_file=None,
            refresh_rate=self.configuration["refresh_rate"],
            verbose=False,
        )

    def start_capture(self):
        self.radar.start()
        t0 = time.time()
        while True:
            try:
                # -- read from serial port
                time.sleep(self.radar.refresh_delay)
                xyzrrt = self.radar.read_serial()
                if xyzrrt is None:
                    continue
                razelrrt = xyzrrt.copy()
                razelrrt[:, :3] = matrix_cartesian_to_spherical(xyzrrt[:, :3])

                # -- send across comms channel
                timestamp = round(time.time() - t0, 9)
                msg = {
                    "timestamp": timestamp,
                    "frame": self.frame,
                    "identifier": self.identifier,
                    "extrinsics": [0, 0, 0, 0, 0],
                }
                self.backend.send_array(razelrrt, msg, False)
                if self.debug:
                    self.print(
                        f"sending data, frame: {msg['frame']:4d}, timestamp: {msg['timestamp']:.4f}",
                        end="\n",
                    )
                self.frame += 1
            except KeyboardInterrupt:
                self.radar.streamer.stop_serial_stream()
                if self.verbose or self.debug:
                    print("Radar.stream_serial: stopping serial stream")
                break


class Camera(Sensor):
    NAME = "camera-sensor"

    def __init__(
        self,
        context,
        BACKEND_HOST,
        BACKEND_PORT,
        sensor_type,
        configs,
        identifier,
        verbose=False,
        debug=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            context,
            BACKEND_HOST,
            BACKEND_PORT,
            sensor_type,
            configs,
            identifier,
            verbose=verbose,
            debug=debug,
        )

        self.handle = None
        self.streaming = False

    def initialize(self):
        if self.sensor_type == "camera-flir-bfs":

            ## Extract sensor configs for flir bfs
            cam_name = self.configuration.get("name", "NA")
            cam_serial = self.configuration.get("serial", "NA")
            cam_ip = self.configuration.get("ip", "NA")
            cam_width_px = int(self.configuration.get("width_px", 0))
            cam_height_px = int(self.configuration.get("height_px", 0))
            cam_fps = int(self.configuration.get("fps", 0))
            cam_frame_size_bytes = int(self.configuration.get("frame_size_bytes", 0))

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
            fx = 1448  
            fy = 1448 
            u = cam_width_px / 2
            v = cam_height_px / 2
            g = 0
            msg = {
                "timestamp": 0.0,
                "frame": 0,
                "identifier": self.identifier,
                "intrinsics": [fx, fy, g, u, v],
                "channel_order": "rgb",
            }

            self.handle.BeginAcquisition()
            t0 = time.time()
            frame_counter = 0
            while True:
                ptr = self.handle.GetNextImage()
                if ptr.IsIncomplete():
                    continue  # discard image
                timestamp = round(time.time() - t0, 9)  # * 1e-9 # ms
                msg["timestamp"] = timestamp
                msg["frame"] = frame_counter

                # -- Version 1: successfully gets colored image as ndarray
                arr = np.frombuffer(ptr.GetData(), dtype=np.uint8).reshape(
                    self.image_dimensions
                )
                img = cv2.cvtColor(
                    arr, cv2.COLOR_BayerBG2BGR
                )  # np.ndarray with d = (h, w, 3)

                # -- used for calibration...
                # cv2.imwrite("flir-bfs.jpg", img)
                # print("saved image... ending program")
                # break

                # -- resize image before compression
                new_h = int(img.shape[0] / self.configuration["resize_factor"])
                new_w = int(img.shape[1] / self.configuration["resize_factor"])
                img_resized = cv2.resize(
                    img, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
                img = img_resized

                # -- image compression
                success, result = cv2.imencode(
                    ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                if not success:
                    raise RuntimeError("Error compressing image")
                compressed_frame = np.array(result)
                img = np.ascontiguousarray(compressed_frame)

                self.backend.send_array(img, msg, False)
                if self.debug:
                    self.print(
                        f"sending data, frame: {frame_counter:4d}, timestamp: {timestamp:.4f}",
                        end="\n",
                    )
                ptr.Release()
                frame_counter += 1

        elif self.sensor_type == "camera-rpi":
            pass
        else:
            pass

    def start_capture(self):
        print("entered start_capture() ")

        if self.sensor_type == "camera-flir-bfs":
            fx = 1448  
            fy = 1448 
            u = self.image_dimensions[1] / 2
            v = self.image_dimensions[0] / 2
            g = 0
            msg = {
                "timestamp": 0.0,
                "frame": 0,
                "identifier": self.identifier,
                "intrinsics": [fx, fy, g, u, v],
                "channel_order": "rgb",
                "compression": "jpeg",
            }

            #! Method should end here
            self.handle.BeginAcquisition()
            t0 = 0
            frame_counter = 0
            while True:

                ptr = self.handle.GetNextImage()

                timestamp = float(ptr.GetTimeStamp()) * 1e-9  # ms
                if frame_counter == 0:
                    t0 = timestamp
                msg["timestamp"] = round(timestamp - t0, 9)
                print(timestamp)
                msg["frame"] = frame_counter

                arr = np.frombuffer(ptr.GetData(), dtype=np.uint8).reshape(
                    self.image_dimensions
                )
                img = cv2.cvtColor(arr, cv2.COLOR_BayerBG2BGR)  # np.ndarray
                ret, jpeg_buffer = cv2.imencode(
                    ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                )
                if not ret:
                    raise RuntimeError("Error compressing image")
                compressed_frame = np.array(jpeg_buffer)
                img = np.ascontiguousarray(compressed_frame)

                # if not img.flags["C_CONTIGUOUS"]:
                #     img = np.ascontiguousarray(img)

                self.backend.send_array(img, msg)
                if self.verbose:
                    self.print(
                        f"sending data, frame: {frame_counter:4d}, timestamp: {timestamp:.4f}",
                        end="\n",
                    )
                frame_counter += 1

        elif self.sensor_type == "camera-rpi":
            pass
        else:
            pass

    def stop_capture(self):
        pass

    def reconfigure():
        pass


def main(args, configs):

    # -- init sensor class
    context = SerializingContext()
    if "camera" in args.sensor_type:
        SensorClass = Camera
    elif "radar" in args.sensor_type:
        SensorClass = Radar
    else:
        raise NotImplementedError(args.sensor_type)
    sensor = SensorClass(
        context,
        args.host,
        args.backend,
        args.sensor_type,
        configs, 
        configs["name"],
        verbose=args.verbose,
        debug=args.debug,
    )

    # -- initialize and run sensor
    sensor.initialize()
    print("Sensor successfully initialized in sensor.py")
    sensor.start_capture()


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
            "resize_factor": 4,
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
            "resize_factor": 4,
        },
        "camera_3": {
            "name": "camera_jackwhite",
            "type": "camera-rpi",
            "serial": "NA",
            "ip": "192.168.1.2",
            "width_px": "640",
            "height_px": "480",
            "fps": "25",
            "frame_size_bytes": "NA",
            "resize_factor": 4,
        },
        "radar_1": {
            "name": "radar_1",
            "config_file_name": "1443config.cfg",
            "CLI_port": "/dev/ttyACM0",
            "Data_port": "/dev/ttyACM1",
            "refresh_rate": 50.0,
        },
    }

    parser = argparse.ArgumentParser("Initialize a Sensor")
    parser.add_argument(
        "--config",
        choices=list(configs.keys()),
        type=str,
        help="Select the configuration to apply",
    )
    parser.add_argument(
        "--sensor_type",
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
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    sensor_data = configs[args.config]

    main(args, sensor_data)
