#!/usr/bin/env python3

import argparse
import json
import logging

import zmq
from avstack.datastructs import DelayManagedDataBuffer
from avstack.geometry import GlobalOrigin3D
from avstack.modules.perception.detections import DetectionContainerDecoder
from avstack.modules.tracking import tracker2d, tracker3d

from avstack.utils.decorators import profileit

from jumpstreet.utils import BaseClass, config_as_namespace, init_some_end


def init_tracking_model(model):
    if model == "sort":
        tracker = tracker2d.SortTracker2D()
    elif model == "passthrough":
        tracker = tracker2d.PassthroughTracker2D()
    elif model == "BasicRazTracker":
        tracker = tracker3d.BasicRazTracker()
    else:
        raise NotImplementedError(model)
    return tracker


class ObjectTracker(BaseClass):
    NAME = "object-tracker"

    def __init__(
        self,
        context,
        model,
        frontend,
        backend,
        dt_delay=0.1,
        reference=GlobalOrigin3D,
        verbose=False,
        debug=False,
    ) -> None:
        """Set up front and back ends

        Front end: sub
        Back end: pub
        """
        super().__init__(name=self.NAME, identifier=1, verbose=verbose, debug=debug)
        self.frontend = init_some_end(
            self,
            context,
            "frontend",
            zmq.SUB,
            frontend.transport,
            frontend.host,
            frontend.port,
            BIND=frontend.bind,
            subopts=b"detections",
        )
        self.backend = init_some_end(
            self,
            context,
            "backend",
            zmq.PUB,
            backend.transport,
            backend.host,
            backend.port,
            BIND=backend.bind,
        )
        self.n_dets = 0
        self.model = init_tracking_model(model)
        self.dt_delay = dt_delay
        self.t_last_emit = None
        self.detection_buffer = DelayManagedDataBuffer(
            dt_delay=dt_delay, max_size=30, method="event-driven"
        )
        self.reference = reference
        self.poller = zmq.Poller()
        self.poller.register(self.frontend, zmq.POLLIN)

        # -- hack to get unique profile name...
        @profileit(f'profile_tracker.prof', folder='profiles')
        def poll(*args, **kwargs):
            return self._poll(*args, **kwargs)
        self.poll = poll

    def _poll(self):
        sockets = dict(self.poller.poll())

        if self.frontend in sockets:
            # -- get data from frontend
            key, data = self.frontend.recv_multipart()
            detections = json.loads(data.decode(), cls=DetectionContainerDecoder)
            # -- put detections on the buffer
            self.detection_buffer.push(detections)

        # -- process data, if ready
        if self.model is not None:
            detections_dict = self.detection_buffer.emit_one()
            if len(detections_dict) > 0:
                # for now, there can only be one key in the detections
                assert len(detections_dict) == 1
                detections = detections_dict[list(detections_dict.keys())[0]]
                if self.debug:
                    self.print(
                        f"Processing detections frame: {detections.frame:4d}, time: {detections.timestamp:.4f}",
                        end="\n",
                    )
                tracks = self.model(
                    detections=detections,
                    t=detections.timestamp,
                    frame=detections.frame,
                    identifier="tracker-0",
                    platform=self.reference,
                )
                if self.debug:
                    self.print(f"currently maintaining {len(tracks)} tracks", end="\n")
                tracks = str.encode(tracks.encode())

                # -- send data at backend
                self.backend.send_multipart([b"tracks", tracks])


def main(config):
    """Run tracking algorithm"""
    context = zmq.Context.instance()
    tracker = ObjectTracker(
        context,
        config.model,
        config.frontend,
        config.backend,
        verbose=config.verbose,
        debug=config.debug,
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
    parser.add_argument("--config", default="tracking/default.yml")
    args = parser.parse_args()
    config = config_as_namespace(args.config)
    main(config)
