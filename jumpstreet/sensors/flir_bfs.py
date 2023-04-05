import numpy as np
import zmq
import cv2
import PySpin

from context import SerializingContext
from sensor import Sensor, interpolate_jpg, compress_to_jpg


"""
Author: Nate Zelter
Date: April 2023
Mobile Sensor Fusion Platform (MSFP)

This module contains the Sensor class, which is an abstract class for acquiring data from different types of sensors.
The Sensor class defines methods for initializing the sensor, starting and stopping data capture, and reconfiguring the sensor.
Other classes can inherit from the Sensor class to implement specific sensor types.

Arguments:
- context (zmq.Context): ZeroMQ context object for communication.
- identifier (str): Unique identifier (name) for the class instance.
- configs (dict): Dictionary of sensor configurations, such as camera settings.
- host (str): Host IP address for communication. Default is "127.0.0.1" (equiv to "localhost" and "*").
- backend (int): Backend port number for communication. Default is DEFAULT_BACKEND_PORT.
- backend_other (int): Additional backend port number for communication. Default is None.
- verbose (bool): Whether to print messages to the console. Default is False.

Instantiate as: 
- sensor = FlirBfs(context, identifier, configs, 
                   host="127.0.0.1", backend=DEFAULT_BACKEND_PORT, 
                   backend_other=None, verbose=False)
"""

DEFAULT_BACKEND_PORT = 6551
STOP_KEY = 'q'


class FlirBfs(Sensor):
    """
    FLIR BFS camera implementation of Sensor class.
    """

    def __init__(self, context, identifier, configs, host="127.0.0.1", backend=DEFAULT_BACKEND_PORT, 
                 backend_other=None, verbose=False, *args, **kwargs):
        super().__init__(context=context, identifier=identifier, type="camera-flir-bfs", configs=configs, 
                         host=host, backend=backend, backend_other=backend_other, verbose=verbose, *args, **kwargs)

        self.handle = None
        self.image_dimensions = None


    def initialize(self):
        """
        Initialize the FLIR BFS camera by setting properties and connecting to the camera handle.
        """

        super().initialize() # calls Sensor.initialize(), may throw a TypeError

        ## -- extract camera properties from configs -- ##
        try:
            cam_name = self.configs.get("name")
            cam_serial = self.configs.get("serial")
            cam_width_px = int(self.configs.get("width_px"))
            cam_height_px = int(self.configs.get("height_px"))
            cam_fps = int(self.configs.get("fps"))
        except:
            raise RuntimeError("Unable to extract camera properties from configs")

        ## -- set additional sensor properties -- ##
        self.image_dimensions = (cam_height_px, cam_width_px)

        ## -- connect to FLIR camera, sensor instance as self.handle -- ##
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        self.handle = cam_list.GetBySerial(cam_serial)
        try:
            self.handle.Init()
            print(f"Successfully connected to {cam_name} via serial number")
        except:
            msg = f"Unable to connect to {cam_name} via serial number"
            raise RuntimeError(msg)

        ## -- set camera properties from extracted configs-- ##
        self.handle.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        self.handle.Width.SetValue(cam_width_px)
        self.handle.Height.SetValue(cam_height_px)
        self.handle.AcquisitionFrameRateEnable.SetValue(True)
        self.handle.AcquisitionFrameRate.SetValue(cam_fps) # Max is 23 fps for FLIR BFS

        ''' ------------------------------------------------------------
        FUNCTION SHOULD END HERE
        ------------------------------------------------------------ '''

        ## -- intrinsics -- ##
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

        ## -- start acquisition (process each frame and publish it) -- ##
        self.handle.BeginAcquisition()
        t0 = 0
        frame_counter = 0
        while True:

            ## -- get image valid image pointer -- ##
            ptr = self.handle.GetNextImage()
            if ptr.IsIncomplete():
                continue # discard image

            ## -- get timestamp and frame number -- ##
            ts_raw = float(ptr.GetTimeStamp())
            timestamp = ts_raw * 1e-9 # ms
            if frame_counter == 0:
                t0 = timestamp
            msg["timestamp"] = round(timestamp - t0, 9)
            msg["frame"] = frame_counter

            # -- convert image to numpy array and process -- ##
            arr = np.frombuffer(ptr.GetData(), dtype=np.uint8).reshape(self.image_dimensions)
            img_raw = cv2.cvtColor(arr, cv2.COLOR_BayerBG2BGR)  # np.ndarray with d = (h, w, 3)
            img_interpolated = interpolate_jpg(img_raw, self.resize_factor) 
            img_compressed = compress_to_jpg(img_interpolated, 80) # TODO change 80 to self.quality  
            img = np.ascontiguousarray(img_compressed)
        
            ## -- publish image with ZMQ -- ##
            self.backend.send_array(img, msg, False)
            if self.verbose:
                self.print(f"sent data, frame: {frame_counter:4d}, timestamp: {timestamp:.4f}", end="\n")

            ptr.Release()
            frame_counter += 1


    def start_capture(self):
        """
        Start capturing data from the FLIR BFS camera.
        """

        ## -- intrinsics -- ##
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
        }

        super().start_capture()
        image_dimensions = self.image_dimensions

        ## -- start acquisition: process each frame and publish it -- ##
        self.handle.BeginAcquisition()
        t0 = 0
        frame_counter = 0
        while True:

            ## -- get image valid image pointer -- ##
            ptr = self.handle.GetNextImage()
            if ptr.IsIncomplete():
                continue # discard image

            ## -- get timestamp and frame number -- ##
            ts_raw = float(ptr.GetTimeStamp())
            timestamp = ts_raw * 1e-9 # ms
            if frame_counter == 0:
                t0 = timestamp
            msg["timestamp"] = round(timestamp - t0, 9)
            msg["frame"] = frame_counter

            # -- convert image to numpy array and process -- ##
            arr = np.frombuffer(ptr.GetData(), dtype=np.uint8).reshape(self.image_dimensions)
            img_raw = cv2.cvtColor(arr, cv2.COLOR_BayerBG2BGR)  # np.ndarray with d = (h, w, 3)
            img_interpolated = interpolate_jpg(img_raw, self.resize_factor) 
            img_compressed = compress_to_jpg(img_interpolated, 80) # TODO change 80 to self.quality  
            img = np.ascontiguousarray(img_compressed)
        
            ## -- publish image with ZMQ -- ##
            self.backend.send_array(img, msg, False)
            if self.verbose:
                self.print(f"sent data, frame: {frame_counter:4d}, timestamp: {timestamp:.4f}", end="\n")

            ptr.Release()
            frame_counter += 1

        self.handle.EndAcquisition()
        self.handle.DeInit()


    def stop_capture(self):
        raise NotImplementedError


    def reconfigure(self, configs):
        raise NotImplementedError




def main(args):
    # TODO implement main method to take args from controller
    pass




if __name__ == "__main__":
    ## -- load configuration from json (this will later be done by controller) -- ##
    configs = {
        "camera_1": {
            "name": "camera_1",
            "type": "FLIR-BFS-50S50C",
            "serial": "22395929",
            "ip": "192.168.1.1",
            "width_px": "2448",
            "height_px": "2048",
            "fps": "10",
            "frame_size_bytes": "307200",
        }
    }

    ## -- create ZMQ context and instantiate flir_bfs object -- ##
    context = zmq.Context()
    flir_bfs = FlirBfs(
        context=context,
        identifier=configs["camera_1"]["name"],
        configs=configs["camera_1"],
        host="127.0.0.1",
        backend=DEFAULT_BACKEND_PORT,
        backend_other=None,
        verbose=False,
    )

    ## -- initialize and start capture -- ##
    flir_bfs.initialize()
    if flir_bfs.handle is None:
        raise RuntimeError("Unable to initialize FLIR BFS camera")
    flir_bfs.print("Initialized FLIR BFS camera, starting capture...")
    flir_bfs.start_capture()

    """
    this wont work because there is an infinite loop in start_capture()
    """
    while True:
        accept = input(f"Press '{STOP_KEY}' to stop capture: ")
        if accept == STOP_KEY:
            print("Received STOP_KEY signal")
            break

    ## -- stop capture and close -- ##
    flir_bfs.print("Stopping capture...")
    flir_bfs.stop_capture()
    flir_bfs.close()