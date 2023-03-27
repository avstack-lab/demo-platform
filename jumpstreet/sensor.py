#!/usr/bin/env python3

"""
Author: Nate Zelter
Date: March 2023

"""

import argparse
import glob
import logging
import multiprocessing
import os
import time
import json
import numpy as np
import cv2

import zmq
import PySpin

from context import SerializingContext
from utils import BaseClass, init_some_end, send_array_pubsub, send_jpg_pubsub

STOP_KEY = 'q'
DEFAULT_BACKEND_PORT = 6551
ACCEPTABLE_SENSOR_TYPES = [
    'camera-flir-bfs',
    'camera-rpi'
]

class Sensor(BaseClass):
    """
        Sensor ZMQ Node, device agnostic
        Pattern: Data aquisition --> context.socket(zmq.PUB)
    """

    NAME = "sensor"

    def __init__(self,
                 context, 
                 identifier, 
                 type,
                 configs, 
                 backend_port,
                 host="*",
                 verbose=False,
                 *args,
                 **kwargs) -> None:
        super().__init__(self.NAME, identifier)

        if type in ACCEPTABLE_SENSOR_TYPES:
            self.type = type
        else:
            raise NotImplementedError(f'Unacceptable sensor type: {type}')

        self.configs = configs

        self.backend = init_some_end(
            self, 
            context, 
            "backend", 
            zmq.PUB, 
            host, 
            backend_port, 
            BIND=True)
        
        self.verbose = verbose



    def initialize(self):
        if self.type == 'camera-flir-bfs':

            ## Extract sensor configs for flir bfs
            cam_name = self.configs.get('name', 'NA')
            cam_serial = self.configs.get('serial', 'NA')
            cam_ip = self.configs.get('ip', 'NA')  
            cam_width_px = int(self.configs.get('width_px', 0))  
            cam_height_px = int(self.configs.get('height_px', 0))  
            cam_fps = int(self.configs.get('fps', 0))
            cam_frame_size_bytes = int(self.configs.get('frame_size_bytes', 0))

            ## Connect to camera
            system = PySpin.System.GetInstance()
            cam_list = system.GetCameras()
            cam = cam_list.GetBySerial(cam_serial)
            try:
                cam.Init()
                print(f'Successfully connected to {cam_name} via serial number')
            except:
                raise RuntimeError(f"Unable to connect to {cam_name} via serial number")

            ## Set the camera properties here
            cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            cam.Width.SetValue(cam_width_px)
            cam.Height.SetValue(cam_height_px)
            cam.AcquisitionFrameRateEnable.SetValue(True) #enable changes to FPS
            cam.AcquisitionFrameRate.SetValue(cam_fps) # max is 24fps for FLIR BFS

            # new (start)
            image_dimensions = (cam_height_px, cam_width_px)
            cam.BeginAcquisition()
            for i in range(100):
                ptr = cam.GetNextImage()
                arr = np.frombuffer(ptr.GetData(), dtype=np.uint8).reshape(image_dimensions)
                img = cv2.cvtColor(arr, cv2.COLOR_BayerBG2BGR)
                # img = np.ascontiguousarray(img)
                array = img.tobytes()
                msg = b'sample'
                # send_array_pubsub(self.backend, msg, array)
                # self.backend.send_array_pubsub(array, msg, False)
                print(len(img))
                image_show = cv2.resize(img, None, fx=0.25, fy=0.25)
                cv2.imshow(f"Press {STOP_KEY} to quit", image_show)
                key = cv2.waitKey(30)
                if key == ord(STOP_KEY):
                    print('Received STOP_KEY signal')
                    ptr.Release()
                    # cam.EndAcquisition()
                    break
            # new (end)


            # self.handle = cam


        elif self.type == 'camera-rpi':
            pass
        else:
            pass

    def start_capture(self):
        if self.type == 'camera-flir-bfs':
            self.handle.BeginAcquisition()
        elif self.type == 'camera-rpi':
            pass
        else:
            pass


    def stop_capture(self):
        pass

    def publish(self):
        if self.type == 'camera-flir-bfs':
            image_ptr = self.handle.GetNextImage()
            # image_dimensions = (int(self.configs.get('height_px', 0)), int(self.configs.get('width_px', 0)))
            # image_array = np.frombuffer(image_ptr.GetData(), dtype=np.uint8).reshape(image_dimensions)
            image_array = np.frombuffer(image_ptr.GetData(), dtype=np.uint8)
            msg = b''
            self.backend.send_array(image_array, msg, copy=False)
            image_ptr.Release()

        elif self.type == 'camera-rpi':
            pass
        else:
            pass

    
    def reconfigure():
        pass




def main(args, configs):

    ### Instantiate Sensor and configure device ###
    context = SerializingContext()
    sensor = Sensor(
        context,  
        configs['name'],
        args.type,
        configs,
        args.backend
    )
    print("Sensor successfully created in sensor.py")

    sensor.initialize()
    print("Sensor successfully initialized in sensor.py")

    # sensor.start_capture()
    # print("Sensor successfully initiated capture sequence, starting to publish()")
    # for i in range(100):
    #     sensor.publish()



    # ### Publish sensor data (loop) ###
    # try:
    #     while True:
    #         ## --- send data
    #         msg = b'<data to send>'
    #         print(msg)
    #         time.sleep(1)
    #         # publisher1.send_multipart([b"A", b"Hello from node 1!"])


    # except Exception as e:
    #     logging.warning(e, exc_info=True)
    # finally:
    #     sensor.close()
    #     print("Sensor successfully closed in sensor.py")



if __name__ == "__main__":

    ## -- load configuration from json (this will later be done by controller)
    configs = {
    'camera_1': {
        'name': 'camera_1',
        'type': 'FLIR-BFS-50S50C',
        'serial': '22395929',
        'ip': '192.168.1.1',
        'width_px': '2448',
        'height_px': '2048',
        'fps': '10',
        'frame_size_bytes': '307200'  
        },
    'camera_2': {
        'name': 'camera_2',
        'type': 'FLIR-BFS-50S50C',
        'serial': '22395953',
        'ip': '192.168.1.2',
        'width_px': '2448',
        'height_px': '2048',
        'fps': '10',
        'frame_size_bytes': '307200'  
        }
    }

    parser = argparse.ArgumentParser("Initialize a Sensor")
    parser.add_argument(
        "--type", 
        choices=ACCEPTABLE_SENSOR_TYPES,
        type=str, 
        default="camera", # NEED TO CHANGE DEFAULT TO Null
        help="Selection of sensor type")
    parser.add_argument(
        "--backend", 
        type=int, 
        default=DEFAULT_BACKEND_PORT, 
        help="Backend port number (PUB)")
    parser.add_argument(
        "--backend_other", 
        type=int, 
        help="Extra backend port (used only in select classes)")
    parser.add_argument(
        "--verbose", 
        action="store_true") # unsure if --verbose is needed
    
    args = parser.parse_args()


    main(args, configs['camera_1'])