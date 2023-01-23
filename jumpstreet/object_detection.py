#!/usr/bin/env python3

import argparse
import logging
import multiprocessing
from time import sleep

import zmq

from jumpstreet.context import SerializingContext
from jumpstreet.utils import BaseClass, init_some_end


class ObjectDetection(BaseClass):
    NAME = "object-detector"

    def __init__(
        self, context, IN_HOST, IN_PORT, OUT_HOST, OUT_PORT, OUT_BIND, identifier
    ) -> None:
        """Set up front and back ends

        Front end: req
        Back end: pub
        """
        super().__init__(name=self.NAME, identifier=identifier)
        self.frontend = init_some_end(
            self, context, "frontend", zmq.REQ, IN_HOST, IN_PORT, BIND=False
        )
        self.backend = init_some_end(
            self, context, "backend", zmq.PUB, OUT_HOST, OUT_PORT, BIND=OUT_BIND
        )
        self.n_imgs = 0

        # -- ready to go (need this!)
        self.frontend.send(b"READY")
        self.print(f"ready to start", end="\n")

    def poll(self):
        """Poll for messages

        Address is the place to send back data
        """
        # -- get data from frontend
        address, metadata, array = self.frontend.recv_array_multipart(copy=False)
        # -- process data
        detections = b"no detections yet"
        self.n_imgs += 1
        self.print(f"received image - total is {self.n_imgs}", end="\n")
        # -- send data at backend
        self.backend.send_multipart([b"detections", detections])
        # -- say we're ready for more
        self.frontend.send_multipart([address, b"", b"OK"])


def start_worker(task, *args):
    """Starting a worker using multiproc"""
    process = multiprocessing.Process(target=task, args=args)
    process.daemon = True
    process.start()
    return process


def main_single(IN_HOST, IN_PORT, OUT_HOST, OUT_PORT, OUT_BIND, identifier):
    """Runs polling on a single worker"""
    context = SerializingContext()
    detector = ObjectDetection(
        context, IN_HOST, IN_PORT, OUT_HOST, OUT_PORT, OUT_BIND, identifier
    )
    try:
        while True:
            detector.poll()
    except Exception as e:
        logging.warning(e, exc_info=True)
    finally:
        detector.close()


def main(args):
    """Run object detection workers"""
    procs = []
    for i in range(args.nworkers):
        proc = start_worker(
            main_single,
            args.in_host,
            args.in_port,
            args.out_host,
            args.out_port,
            args.out_bind,
            i,
        )
        procs.append(proc)

    while True:
        sleep(1)
        any_alive = False
        for proc in procs:
            proc.join(timeout=0)
            if proc.is_alive():
                any_alive = True
                break
        if not any_alive:
            logging.warning("exiting because no more processes alive")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Initialize object detection workers")
    parser.add_argument(
        "-n", "--nworkers", type=int, default=2, help="Number of workers"
    )
    parser.add_argument(
        "--in_host", default="localhost", type=str, help="Hostname to connect to"
    )
    parser.add_argument(
        "--in_port", default=5556, type=int, help="Port to connect to server/broker"
    )
    parser.add_argument(
        "--out_host",
        default="localhost",
        type=str,
        help="Hostname to connect output to",
    )
    parser.add_argument(
        "--out_port", default=5557, type=int, help="Port to connect output data to"
    )
    parser.add_argument(
        "--out_bind",
        action="store_true",
        help="Whether or not the output connection binds here",
    )

    args = parser.parse_args()
    main(args)
