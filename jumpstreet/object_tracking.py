#!/usr/bin/env python3

import argparse
import logging
import zmq
from time import sleep
from jumpstreet.utils import init_some_end, BaseClass


class ObjectTracker(BaseClass):
    NAME = 'object-tracker'

    def __init__(self, context, IN_HOST, IN_PORT, OUT_HOST, OUT_PORT, IN_BIND=True, OUT_BIND=True) -> None:
        """Set up front and back ends

        Front end: sub
        Back end: pub
        """
        super().__init__(name=self.NAME, identifier=1)
        self.frontend = init_some_end(self, context, 'frontend', zmq.SUB, IN_HOST, IN_PORT, BIND=IN_BIND, subopts=b"detections")
        self.backend  = init_some_end(self, context, 'backend',  zmq.PUB, OUT_HOST, OUT_PORT, BIND=OUT_BIND)
        self.n_dets = 0

    def poll(self):
        # -- get data from frontend
        key, data = self.frontend.recv_multipart()
        # -- process data
        tracks = b"no tracks yet"
        self.n_dets += 1
        self.print(f'received detections - total is {self.n_dets}', end='\n')
        # -- send data at backend
        self.backend.send_multipart([b"tracks", tracks])


def main(args):
    """Run tracking algorithm"""
    context = zmq.Context.instance()
    tracker = ObjectTracker(context, args.in_host, args.in_port, args.out_host, 
        args.out_port, args.in_bind, args.out_bind)

    try:
        while True:
            tracker.poll()
    except Exception as e:
        logging.warning(e, exc_info=True)
    finally:
        tracker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Initialize object detection workers')
    parser.add_argument('--in_host', default='localhost', type=str, help='Hostname to connect to')
    parser.add_argument('--in_port', default=5557, type=int, help='Port to connect to server/broker')
    parser.add_argument('--in_bind', action="store_true", help='Whether or not the input connection binds here')
    parser.add_argument('--out_host', default='localhost', type=str, help='Hostname to connect output to')
    parser.add_argument('--out_port', default=5558, type=int, help='Port to connect output data to')
    parser.add_argument('--out_bind', action="store_true", help='Whether or not the output connection binds here')

    args = parser.parse_args()
    main(args)