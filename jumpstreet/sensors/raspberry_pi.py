import io
import time  # just being used a placeholder for now

import cv2
import numpy as np
import picamera
import zmq
from context import SerializingContext
from sensor import Sensor, compress_to_jpg, interpolate_jpg


DEFAULT_BACKEND_PORT = 6552
STOP_KEY = "q"


class RaspberryPi(Sensor):
    """
    Raspberry Pi camera implementation of Sensor class.
    """

    def __init__(
        self,
        context,
        identifier,
        configs,
        host="127.0.0.1",
        backend=DEFAULT_BACKEND_PORT,
        backend_other=None,
        verbose=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            context=context,
            identifier=identifier,
            type="camera-raspberry-pi",
            configs=configs,
            host=host,
            backend=backend,
            backend_other=backend_other,
            verbose=verbose,
            *args,
            **kwargs,
        )

        # Add any additional initialization code specific to Raspberry Pi camera here
        self.handle = None
        self.image_dimensions = None

    def initialize(self):
        """
        Initialize the Raspberry Pi camera by setting properties and connecting to the camera handle.
        """

        super().initialize()  # calls Sensor.initialize(), may throw a TypeError
        pass

        ## -- extract camera properties from configs -- ##
        try:
            cam_width_px = int(self.configs.get("width_px"))
            cam_height_px = int(self.configs.get("height_px"))
            cam_fps = int(self.configs.get("fps"))
            cam_ip = self.configs.get("ip")
            cam_serial = self.configs.get("serial")
        except:
            raise RuntimeError("Unable to extract camera properties from configs")

        ## -- set additional sensor properties -- ##
        self.image_dimensions = (cam_height_px, cam_width_px)

        ## -- initialize Raspberry Pi camera -- ##
        # self.handle = picamera.PiCamera()
        self.handle = picamera.PiCamera(hostname=cam_ip)
        # self.handle = picamera.PiCamera(camera_num=cam_serial)
        self.handle.resolution = (cam_width_px, cam_height_px)
        self.handle.framerate = cam_fps

    def stop_capture(self):
        """
        Stop capturing data from the Raspberry Pi camera.
        """
        # TODO Add code to stop capturing data from the Raspberry Pi camera, similar to the FlirBfs implementation
        pass

    def start_capture(self):
        """
        Start capturing data from the Raspberry Pi camera.
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

        stream = io.BytesIO()
        t0 = 0
        frame_counter = 0
        for frame in self.handle.capture_continuous(
            stream, format="jpeg", use_video_port=True
        ):
            timestamp = time.monotonic()  # seconds
            if frame_counter == 0:
                t0 = timestamp
            msg["timestamp"] = round(timestamp - t0, 9)
            msg["frame"] = frame_counter

            ## -- convert image to numpy array and process -- ##
            img_raw = np.frombuffer(frame.getvalue(), dtype=np.uint8)
            img_raw = cv2.imdecode(
                img_raw, cv2.IMREAD_COLOR
            )  # np.ndarray with d = (h, w, 3)
            img_interpolated = interpolate_jpg(img_raw, self.resize_factor)
            img_compressed = compress_to_jpg(
                img_interpolated, 80
            )  # TODO change 80 to self.quality
            img = np.ascontiguousarray(img_compressed)

            ## -- publish image with ZMQ -- ##
            self.backend.send_array(img, msg, False)
            if self.verbose:
                self.print(
                    f"sent data, frame: {frame_counter:4d}, timestamp: {timestamp:.4f}",
                    end="\n",
                )

            ## -- check if stop key was pressed -- ##
            accept = input(f"Press {STOP_KEY} to quit: ")
            if accept == STOP_KEY:
                break

            frame_counter += 1

        self.handle.close()
        self.stop_capture()

    def reconfigure(self, configs):
        """
        Reconfigure the Raspberry Pi camera with new settings.
        """

        # TODO Add code to reconfigure the Raspberry Pi camera with new settings, similar to the FlirBfs implementation
        pass


def main(args):

    context = SerializingContext()
    rpi = RaspberryPi(context, "rpi", {"width_px": 640, "height_px": 480, "fps": 30})
    rpi.initialize()
    rpi.start_capture()


if __name__ == "__main__":
    # TODO Add code to parse command line arguments, similar to the FlirBfs implementation
    # TODO Add code to initialize the Raspberry Pi camera, similar to the FlirBfs implementation
    # TODO Add code to start capturing data from the Raspberry Pi camera, similar to the FlirBfs implementation

    configs = {
        "name": "camera_jackwhite",
        "type": "camera-rpi",
        "serial": "NA",
        "ip": "192.168.1.2",
        "width_px": "640",
        "height_px": "480",
        "fps": "25",
        "frame_size_bytes": "NA",
    }

    args = {}
    main(args)
    pass
