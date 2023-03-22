#!/usr/bin/env python3

"""
Author: Nate Zelter
Date: February 2023
"""

import argparse
import glob
import logging
import multiprocessing
import os
import time

import numpy as np
import zmq

# from jumpstreet.context import SerializingContext
from context import SerializingContext
from utils import BaseClass, init_some_end

DEFAULT_FRONTEND_PORT = 6550
DEFAULT_BACKEND_PORT = 6551


class SensorDataReceiver(BaseClass):
    """
        Subscribes to sensor data publisher
        ZMQ pattern: SUB --> Router
    """

    NAME = "sensor-data-receiver"

    def __init__(self, 
                 context, 
                 type,
                 FRONTEND=DEFAULT_FRONTEND_PORT, 
                 BACKEND=DEFAULT_BACKEND_PORT, 
                 identifier=0, 
                 verbose=False, 
                 *args, 
                 **kwargs) -> None:
        super().__init__(self.NAME, identifier)
        
        self.type = type
        self.verbose = verbose

        self.frontend = init_some_end(
            self, 
            context, 
            "frontend", 
            zmq.SUB, 
            "*", 
            FRONTEND, 
            BIND=True, 
            subopts=b"")

        self.backend = init_some_end(
            self, 
            context, 
            "backend", 
            zmq.ROUTER, 
            "*", 
            BACKEND, 
            BIND=True)

        self.print(f"initializing frontend poller...", end="")
        self.poller = zmq.Poller()
        self.poller.register(self.frontend, zmq.POLLIN) # register poller with SUB frontend
        print("done")


    def poll(self):
        print("starting poll()")
        socks = dict(self.poller.poll(timeout=500)) # timeout in ms

        if self.frontend in socks and socks[self.frontend] == zmq.POLLIN:
            print("polling")
            msg = self.frontend.recv_multipart()
            response = process_image_data(msg)
            self.backend.send_multipart(response)  
        print("ending poll()")

    def close(self):
        self.frontend.close()
        self.backend.close()
        self.context.term()


def process_image_data(msg):
    # Process the received image data and return the response
    # Replace this with to-be-implemented logic
    print("Received image data:", msg)
    return [b"", b"ACK"]    


def main(args):
    context = SerializingContext()
    sdr = SensorDataReceiver(
        context, 
        type=args.type,
        verbose=args.verbose, 
        FRONTEND=args.frontend, 
        BACKEND=args.backend,
        BACKEND_OTHER=args.backend_other,
        subopts=None
    )
    print("SensorDataReceiver successfully created in receiver.py")
    try:
        while True:
            sdr.poll()
    except Exception as e:
        logging.warning(e, exc_info=True)
    finally:
        sdr.close()
        print("SensorDataReceiver successfully closed in receiver.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Initialize a SensorDataReceiver")
    parser.add_argument(
        "--type", 
        choices=["camera", "radar", "lidar"],
        type=str, 
        default="camera", # NEED TO CHANGE DEFAULT TO Null
        help="Selection of sensor type")
    parser.add_argument(
        "--frontend", 
        type=int, 
        default=DEFAULT_FRONTEND_PORT, 
        help="Frontend port number (sensor)")
    parser.add_argument(
        "--backend", 
        type=int, 
        default=DEFAULT_BACKEND_PORT, 
        help="Backend port number (router)")
    parser.add_argument(
        "--backend_other", 
        type=int, 
        help="Extra backend port (used only in select classes)")
    parser.add_argument(
        "--verbose", 
        action="store_true") # unsure if --verbose is needed
    
    args = parser.parse_args()
    main(args)
