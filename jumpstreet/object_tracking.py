#!/usr/bin/env python3

import argparse
import logging

import zmq
from avstack.modules.perception.detections import get_data_container_from_line
from avstack.modules.tracking import tracker2d
from avstack.modules.tracking.tracks import format_data_container_as_string
from avstack.datastructs import DelayManagedDataBuffer

from jumpstreet.utils import BaseClass, TimeMonitor, init_some_end


def init_tracking_model(model, framerate=30):
    if model == "sort":
        tracker = tracker2d.SortTracker2D(framerate=framerate)
    elif model == "passthrough":
        tracker = tracker2d.PassthroughTracker2D(framerate=framerate)
    else:
        raise NotImplementedError(model)
    return tracker


class ObjectTracker(BaseClass):
    NAME = "object-tracker"

    def __init__(
        self,
        context,
        model,
        IN_HOST,
        IN_PORT,
        OUT_HOST,
        OUT_PORT,
        IN_BIND=True,
        OUT_BIND=True,
        dt_delay=0.1,
        verbose=False,
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
        self.model = init_tracking_model(model)
        self.dt_delay = dt_delay
        self.t_last_emit = None
        self.detection_buffer = DelayManagedDataBuffer(dt_delay=dt_delay, max_size=30, method='event-driven')

    def poll(self):
        # -- get data from frontend
        key, data = self.frontend.recv_multipart()
        detections = get_data_container_from_line(data.decode())

        # -- put detections on the buffer
        self.detection_buffer.push(detections)

        # -- process data, if ready
        if self.model is not None:
            detections_dict = self.detection_buffer.emit_one()
            if len(detections_dict) > 0:
                # for now, there can only be one key in the detections
                assert len(detections_dict) == 1
                detections = detections_dict[list(detections_dict.keys())[0]]
                tracks = self.model(
                    detections,
                    t=detections.timestamp,
                    frame=detections.frame,
                    identifier="tracker-0",
                )
                if self.verbose:
                    self.print(f"currently maintaining {len(tracks)} tracks", end="\n")
                tracks = format_data_container_as_string(tracks).encode()

                # -- send data at backend
                self.backend.send_multipart([b"tracks", tracks])


def main(args):
    """Run tracking algorithm"""
    context = zmq.Context.instance()
    tracker = ObjectTracker(
        context,
        args.model,
        args.in_host,
        args.in_port,
        args.out_host,
        args.out_port,
        args.in_bind,
        args.out_bind,
        verbose=args.verbose,
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
        "--model",
        default="sort",
        choices=["passthrough", "sort"],
        help="Tracking model selection",
    )
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
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    main(args)
