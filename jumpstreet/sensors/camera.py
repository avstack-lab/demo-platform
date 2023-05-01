import time
import cv2
import logging
import numpy as np
import PySpin
from .base import Sensor
from avapi import get_scene_manager
from avstack.calibration import CameraCalibration
from avstack.sensors import ImageData

STOP_KEY = "q"
img_exts = [".jpg", ".jpeg", ".png", ".tiff"]


class NearRealTimeImageLoader:
    """Loads images at nearly the correct rate

    It is expected that this will perform the necessary sleep
    process to enable near-correct-time sending
    """

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.interval = 1.0 / dataset.framerate
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
        img = self.dataset.get_image(self.dataset.frames[self.i_next_img], "main_camera")
        self.counter += 1
        self.i_next_img = (self.i_next_img + 1) % len(self.dataset)
        t_post = time.time()
        if self.t0 is None:
            self.t0 = t_post
        self.dt_last_load = t_post - t_pre_2
        self.next_target_send = self.t0 + self.counter * self.interval
        return img


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

    def _send_image_data(self, img, ts, frame):
        array = img.data
        channel_order = img.calibration.channel_order
        msg = {
            "timestamp": ts,
            "frame": frame,
            "channel_order": channel_order,
            "identifier": self.identifier,
            "calibration": self.calibration.format_as_string(),
        }
        # -- image interpolation
        if (self.config.interpolate is not None) and (self.config.interpolate > 1):
            new_h = int(array.shape[0] // self.config.interpolate)
            new_w = int(array.shape[1] // self.config.interpolate)
            array = cv2.resize(
                array, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
        
        # -- image compression
        success, result = cv2.imencode(".jpg", array, [cv2.IMWRITE_JPEG_QUALITY, self.jpg_compression_pct])
        if not success:
            raise RuntimeError("Error compressing image")
        compressed_frame = np.array(result)
        array = np.ascontiguousarray(compressed_frame)
        self.backend.send_array(array, msg, copy=False)


class ReplayCamera(Camera):
    """A camera that replays a dataset"""
    def __init__(self, context, backend, config, identifier, verbose=False, debug=False) -> None:
        super().__init__(context, backend, config, identifier, verbose, debug)
        SM = get_scene_manager(config.dataset, config.data_dir, config.split)
        SD = SM.get_scene_dataset_by_name(config.scene)
        self.dataset = SD
        self.image_loader = NearRealTimeImageLoader(dataset=SD)

    def send(self):
        img = self.image_loader.load_next()
        ts = self.image_loader.counter / self.dataset.framerate
        frame = self.image_loader.counter
        if self.debug:
            self.print(
                f"sending data, frame: {frame:4d}, timestamp: {ts:.4f}", end="\n"
            )
        self._send_image_data(img, ts, frame)
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
        if "flir" in self.config.model.lower():
            system = PySpin.System.GetInstance()
            cam_list = system.GetCameras()
            self.handle = cam_list.GetBySerial(self.config.serial)
            try:
                self.handle.Init()
                print(f"Successfully connected to {self.config.model} via serial number")
            except:
                raise RuntimeError(f"Unable to connect to {self.config.model} via serial number")

            ## Set the camera properties here
            self.handle.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            self.handle.Width.SetValue(self.config.width)
            self.handle.Height.SetValue(self.config.height)
            self.handle.AcquisitionFrameRateEnable.SetValue(
                True
            )  # enable changes to FPS
            self.handle.AcquisitionFrameRate.SetValue(
                self.config.fps
            )  # max is 24fps for FLIR BFS

            self.image_dimensions = (self.config.height, self.config.width)

            #! Method should end here
            #### --------------------------------------------------------------
            self.handle.BeginAcquisition()
            t0 = time.time()
            frame = -1
            while True:
                ptr = self.handle.GetNextImage()
                if ptr.IsIncomplete():
                    continue  # discard image
                ts = round(time.time() - t0, 9)  # * 1e-9 # ms
                frame += 1

                # -- Version 1: successfully gets colored image as ndarray
                arr = np.frombuffer(ptr.GetData(), dtype=np.uint8).reshape(
                    self.image_dimensions
                )
                arr = cv2.cvtColor(
                    arr, cv2.COLOR_BayerBG2BGR
                )
                img = ImageData(timestamp=ts, frame=frame, source_ID=self.identifier, data=arr, calibration=self.calibration)

                # -- interpolate image before compression
                if self.debug:
                    self.print(
                        f"sending data, frame: {frame:4d}, timestamp: {ts:.4f}", end="\n"
                    )
                self._send_image_data(img, ts, frame)
                self.time_monitor.trigger()
                ptr.Release()
        else:
            raise NotImplementedError(self.config.model)

    def start_capture(self):
        raise RuntimeError('We cannot get here due to some unknown flir handle thing')

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
