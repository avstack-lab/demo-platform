from abc import ABC, abstractmethod
from context import SerializingContext
from utils import BaseClass, init_some_end
import zmq
import cv2
import numpy as np

"""
Author: Nate Zelter
Date: April 2023

This module contains the BaseClass, which is a parent class for other classes 
that need a frontend and backend for communication. The BaseClass defines 
methods for setting and getting the frontend and backend, closing the 
communication channels, and printing messages.

ZMQ Pattern: Sensor --> zmq.PUB as self.backend
                    --> zmq.PUB as self.backend_other

Arguments:
- name (str): Name of the class instance.
- identifier (str): Unique identifier for the class instance.
- verbose (bool): Whether to print messages to the console. Default is False.

Instantiated as:
- sensor = Sensor(context, identifier, type, configs, host, backend, backend_other, verbose=False)
"""

ACCEPTABLE_TYPES = ["camera", "lidar", "radar"]


class Sensor(BaseClass, ABC):
    NAME = "sensor"
    

    def __init__(self, context, identifier, type, configs, host, backend, backend_other, verbose=False, *args, **kwargs):
        super().__init__(self.NAME, identifier, verbose)
        assert type in ACCEPTABLE_TYPES
        self.type = type
        self.configs = configs
        self.backend = init_some_end(self, context, "backend", zmq.PUB, host, backend, BIND=False)
        if backend_other is not None:
            self.backend_other = init_some_end(self, context, "backend_other", zmq.PUB, host, backend_other, BIND=False)

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def start_capture(self):
        pass

    @abstractmethod
    def stop_capture(self):
        pass

    @abstractmethod
    def reconfigure(self):
        pass


""" 
Auxilary Functions for Camera Sensors
"""
def interpolate_jpg(img, resize_factor):
    """
    Interpolate a JPEG image by resizing it using the INTER_AREA interpolation method
    :param img: JPEG image as a numpy ndarray
    :param resize_factor: factor to resize the image by (must be power of 2 to maintain aspect ratio)
    Returns an interpolated image as a numpy ndarray
    """
    new_height = int(img.shape[0] / resize_factor)
    new_width = int(img.shape[1] / resize_factor)
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return img_resized # np.ndarray

def compress_to_jpg(img, quality=95):
    """
    Compression function for JPEG images from OpenCV
    :param img: image as a numpy ndarray
    :param quality: quality of the compressed image (0-100)
    Returns a compressed image as a numpy ndarray
    """
    success, result = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("Error compressing image")
    compressed_frame = np.array(result)
    return compressed_frame # np.ndarray


""" 
Auxilary Functions for Radar Sensors
"""



""" 
Auxilary Functions for Lidar Sensors
"""


