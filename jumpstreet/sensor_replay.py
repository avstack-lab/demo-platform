#!/usr/bin/env python3

import argparse
import logging
import zmq
import glob
import os
import numpy as np
import multiprocessing
import random
from cv2 import imread
from time import sleep
from jumpstreet.utils import init_some_end, BaseClass
from jumpstreet.context import SerializingContext


img_exts = ['.jpg', '.jpeg', '.png', '.tiff']


class SensorDataReplayer(BaseClass):
    """Replays sensor data from a folder"""
    NAME = 'data-replayer'

    def __init__(self, context, HOST, PORT, identifier, send_dir) -> None:
        super().__init__(self.NAME, identifier)
        self.backend = init_some_end(cls=self, context=context, end_type='backend',
            pattern=zmq.REQ, HOST=HOST, PORT=PORT, BIND=False)
        self.images = sorted([img for ext in img_exts for 
            img in glob.glob(os.path.join(send_dir, '*' + ext))])
        self.i_next_img = 0
        self._send_image_data(np.array([]), 'READY')
        ack = self.backend.recv_multipart()
        assert ack[0] == b"OK"
        self.print('confirmed data broker ready', end='\n')

    def send(self):
        # -- load data
        data = imread(self.images[self.i_next_img])

        # -- send data
        self.print('sending data...', end='')
        self._send_image_data(data, f'IMAGE_{self.i_next_img:04d}')
        self.i_next_img = (self.i_next_img + 1) % len(self.images)
        print('done')

        # -- acknowledge
        self.print('waiting for acknowledge...', end='')
        ack = self.backend.recv_multipart()
        assert ack[0] == b"OK"
        print('done')

    def _send_image_data(self, array, msg):
        if array.flags['C_CONTIGUOUS']:
            # if array is already contiguous in memory just send it
            self.backend.send_array(array, msg, copy=False)
        else:
            # else make it contiguous before sending
            array = np.ascontiguousarray(array)
            self.backend.send_array(array, msg, copy=False)


def start_client(task, *args):
    """Starting a client using multiproc"""
    process = multiprocessing.Process(target=task, args=args)
    process.daemon = True
    process.start()


def main_single(HOST, PORT, identifier, send_rate, send_dir):
    """Runs sending on a single client"""
    # context = zmq.Context.instance()
    context = SerializingContext()
    replayer = SensorDataReplayer(context, HOST=HOST, PORT=PORT,
        identifier=identifier, send_dir=send_dir)
    send_dt = 1./send_rate
    try:
        while True:
            replayer.send()
            sleep(send_dt)
    except Exception as e:
        logging.warning(e, exc_info=True)
    finally:
        replayer.close()


def main(args):
    """Run sensor replayer clients"""
    for i in range(args.nclients):
        start_client(main_single, args.host, args.port, i, args.send_rate, args.send_dir)
    while True:
        sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Initialize sensor replayer client')
    parser.add_argument('-n' , '--nclients', type=int, default=1, help='Number of clients')
    parser.add_argument('--host', default='localhost', type=str, help='Hostname to connect to')
    parser.add_argument('--port', default=5555, type=int, help='Port to connect to server/broker')
    parser.add_argument('--send_rate', default=10, type=int, help='Replay rate for sensor data')
    parser.add_argument('--send_dir', type=str, default='./data/TUD-Campus/img1', help='Directory for data replay')

    args = parser.parse_args()
    main(args)