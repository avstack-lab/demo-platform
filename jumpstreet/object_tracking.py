#!/usr/bin/env python3

import argparse
import logging
from time import sleep

import zmq
from jumpstreet.utils import BaseClass, init_some_end

from avstack.modules.perception.detections import get_data_container_from_line
from avstack.modules.tracking.tracker2d import SortTracker2D
from avstack.modules.tracking.tracks import format_data_container_as_string


class ObjectTracker(BaseClass):
    NAME = "object-tracker"

    def __init__(
        self, context, IN_HOST, IN_PORT, OUT_HOST, OUT_PORT, IN_BIND=True, OUT_BIND=True, verbose=False,
    ) -> None:
        """Set up front and back ends

        Front end: sub
        Back end: pub
        """
        super().__init__(name=self.NAME, identifier=1, verbose=verbose)
        self.frontend = init_some_end(
            self,
            context,
            "frontend",
            zmq.SUB,
            IN_HOST,
            IN_PORT,
            BIND=IN_BIND,
            subopts=b"detections",
        )
        self.backend = init_some_end(
            self, context, "backend", zmq.PUB, OUT_HOST, OUT_PORT, BIND=OUT_BIND
        )
        self.n_dets = 0
        self.model = SortTracker2D(framerate=30)

    def poll(self):
        # -- get data from frontend
        key, data = self.frontend.recv_multipart()
        detections = get_data_container_from_line(data.decode())

        # -- process data
        if self.model is not None:
            tracks = self.model(detections, t=detections.timestamp, frame=detections.frame, identifier="tracker-0")
            if self.verbose:
                self.print(f"currently maintaining {len(tracks)} tracks", end="\n")
            tracks = format_data_container_as_string(tracks).encode()
        else:
            tracks = b'No tracks yet'

        # -- send data at backend
        self.backend.send_multipart([b"tracks", tracks])


def main(args):
    """Run tracking algorithm"""
    context = zmq.Context.instance()
    tracker = ObjectTracker(
        context,
        args.in_host,
        args.in_port,
        args.out_host,
        args.out_port,
        args.in_bind,
        args.out_bind,
        args.verbose,
    )

    try:
        while True:
            tracker.poll()
    except Exception as e:
        logging.warning(e, exc_info=True)
    finally:
        tracker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Initialize object detection workers")
    parser.add_argument(
        "--in_host", default="localhost", type=str, help="Hostname to connect to"
    )
    parser.add_argument(
        "--in_port", default=5553, type=int, help="Port to connect to server/broker"
    )
    parser.add_argument(
        "--in_bind",
        action="store_true",
        help="Whether or not the input connection binds here",
    )
    parser.add_argument(
        "--out_host",
        default="localhost",
        type=str,
        help="Hostname to connect output to",
    )
    parser.add_argument(
        "--out_port", default=5554, type=int, help="Port to connect output data to"
    )
    parser.add_argument(
        "--out_bind",
        action="store_true",
        help="Whether or not the output connection binds here",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)
