import time
import cv2
import glob
import os
import logging
import numpy as np
import PySpin
from .base import Sensor
from avstack.calibration import CameraCalibration
from jumpstreet.utils import TimeMonitor


STOP_KEY = "q"
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
        data = cv2.imread(self.image_paths[self.i_next_img])
        channel_order = "bgr"  # most likely loads as BGR since cv2
        self.counter += 1
        self.i_next_img = (self.i_next_img + 1) % len(self.image_paths)
        t_post = time.time()
        if self.t0 is None:
            self.t0 = t_post
        self.dt_last_load = t_post - t_pre_2
        self.next_target_send = self.t0 + self.counter * self.interval
        return data, channel_order


class Camera(Sensor):
    NAME = "camera-sensor"

    def __init__(
        self,
        context,
        backend,
        config,
        identifier,
        verbose=False,
        debug=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            context,
            backend,
            config,
            identifier,
            verbose,
            debug,
        )
        self.frame = 0
        fx = config.calibration.intrinsics.fx
        fy = config.calibration.intrinsics.fy
        u = config.calibration.intrinsics.u
        v = config.calibration.intrinsics.v
        g = config.calibration.intrinsics.g
        P = np.array([[fx,  g, u, 0],
                      [ 0, fy, v, 0],
                      [ 0,  0, 1, 0]])
        img_shape = (config.height, config.width)
        self.calibration = CameraCalibration(self.extrinsics, P, img_shape)
        self.jpg_compression_pct = config.jpg_compression_pct

    def _send_image_data(self, array, ts, frame, channel_order):
        msg = {
            "timestamp": ts,
            "frame": frame,
            "channel_order": channel_order,
            "identifier": self.identifier,
            "calibration": self.calibration.format_as_string(),
        }
        # -- image compression
        success, result = cv2.imencode(".jpg", array, [cv2.IMWRITE_JPEG_QUALITY, self.jpg_compression_pct])
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


class ReplayCamera(Camera):
    """A camera that replays a dataset"""
    def __init__(self, context, backend, config, identifier, verbose=False, debug=False) -> None:
        super().__init__(context, backend, config, identifier, verbose, debug)
        images = sorted(
                [
                    img
                    for ext in img_exts
                    for img in glob.glob(os.path.join(config.data_path, "*" + ext))
                ]
            )
        if len(images) == 0:
            raise RuntimeError(f"No images were found in {config.data_path}!")
        self.fps = config.fps
        self.image_loader = NearRealTimeImageLoader(image_paths=images, rate=self.fps)
        self.time_monitor = TimeMonitor()

    def send(self):
        data, channel_order = self.image_loader.load_next()
        ts = self.image_loader.counter / self.fps
        frame = self.image_loader.counter
        if self.debug:
            self.print(
                f"sending data, frame: {frame:4d}, timestamp: {ts:.4f}", end="\n"
            )
        self._send_image_data(data, ts, frame, channel_order)
        self.time_monitor.trigger()

    def initialize(self):
        pass

    def start_capture(self):
        try:
            while True:
                self.send()
        except Exception as e:
            logging.warning(e, exc_info=True)
        finally:
            self.close()


class PySpinCamera(Camera):
    """A camera based on the PySpin library"""
    def __init__(
        self,
        context,
        backend,
        config,
        identifier,
        verbose=False,
        debug=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            context,
            backend,
            config,
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
