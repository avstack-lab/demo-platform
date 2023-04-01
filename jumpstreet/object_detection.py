#!/usr/bin/env python3

import argparse
import logging
import multiprocessing
from time import sleep
from functools import partial

import numpy as np
import zmq
from avstack.calibration import CameraCalibration
from avstack.geometry import NominalOriginStandard
from avstack.modules.perception.detections import format_data_container_as_string
from avstack.modules.perception.object2dfv import MMDetObjectDetector2D
from avstack.sensors import ImageData

from jumpstreet.context import SerializingContext
from jumpstreet.utils import BaseClass, init_some_end


class ObjectDetector(BaseClass):
    def __init__(
        self,
        context,
        IN_HOST,
        IN_PORT,
        OUT_HOST,
        OUT_PORT,
        OUT_BIND,
        identifier,
        dataset,
        model,
        threshold,
        verbose=False,
    ) -> None:
        """Set up front and back ends

        Front end: req
        Back end: pub
        """
        super().__init__(name=self.NAME, identifier=identifier, verbose=verbose)
        self.frontend = init_some_end(
            self, context, "frontend", zmq.REQ, IN_HOST, IN_PORT, BIND=False
        )
        self.backend = init_some_end(
            self, context, "backend", zmq.PUB, OUT_HOST, OUT_PORT, BIND=OUT_BIND
        )
        self.set_model(dataset, model, threshold)
        self.print("initialized perception model!", end="\n")

        # -- ready to go (need this!)
        self.frontend.send(b"READY")
        self.print(f"ready to start", end="\n")

    def poll(self):
        """Poll for messages

        Address is the place to send back data
        """
        # -- get data from frontend
        address, metadata, array = self.frontend.recv_array_multipart(copy=True)

        # -- process data
        detections = self.detect(metadata, array)
        if detections is not None:
            detections = format_data_container_as_string(detections).encode()
        else:
            detections = b"No detections yet"

        # -- send data at backend
        self.backend.send_multipart([b"detections", detections])
        # -- say we're ready for more
        self.frontend.send_multipart([address, b"", b"OK"])

    def set_model(self):
        raise NotImplementedError
    
    def detect(self):
        raise NotImplementedError


class ImageObjectDetector(ObjectDetector):
    NAME = "image-detector"

    def __init__(
        self,
        context,
        IN_HOST,
        IN_PORT,
        OUT_HOST,
        OUT_PORT,
        OUT_BIND,
        identifier,
        dataset='coco-person',
        model='fasterrcnn',
        threshold=0.5,
        verbose=False,
    ) -> None:
        super().__init__(context, IN_HOST, IN_PORT, OUT_HOST, OUT_PORT, OUT_BIND, identifier, dataset, model, threshold, verbose)

    def set_model(self, dataset, model, threshold):
        # -- set up perception model
        if model == "fasterrcnn":
            try:
                self.model = MMDetObjectDetector2D(
                    dataset=dataset, model=model, threshold=threshold, gpu=0
                )
            except RuntimeError as e:
                if "CUDA error: out of memory" in str(e):
                    self.model = MMDetObjectDetector2D(
                        dataset=dataset, model=model, threshold=threshold, gpu=1
                    )
                else:
                    raise e
        elif model in ["none", None]:
            logging.warning("Not running true object detection")
            self.model = None
        else:
            raise NotImplementedError(model)
    
    def detect(self, metadata, array):
        if self.model is not None:
            if metadata["msg"]["channel_order"].lower() == "rgb":
                is_rgb = True
            elif metadata["msg"]["channel_order"].lower() == "bgr":
                is_rgb = False
            else:
                raise NotImplementedError(metadata["msg"]["channel_order"])
            timestamp = metadata["msg"]["timestamp"]
            frame = metadata["msg"]["frame"]
            if self.verbose:
                self.print(f'Image frame: {frame:4d}, timestamp: {timestamp:.4f}', end='\n')
            identifier = metadata["msg"]["identifier"]
            a, b, g, u, v = metadata["msg"]["intrinsics"]
            P = np.array([[a, g, u, 0], [0, b, v, 0], [0, 0, 1, 0]])
            calib = CameraCalibration(NominalOriginStandard, P, metadata["shape"])
            image = ImageData(
                timestamp=timestamp,
                frame=frame,
                source_ID=identifier,
                source_name="camera",
                data=np.reshape(array, metadata["shape"]),
                calibration=calib,
            )
            
            # -- process data
            detections = self.model(
                image, identifier=metadata["msg"]["identifier"], is_rgb=is_rgb
            )
        else:
            detections = None
        return detections
    

class RadarObjectDetector(ObjectDetector):
    NAME = "radar-detector"
    def set_model(self, dataset, model, threshold):
        raise NotImplementedError


def start_worker(task, *args, **kwargs):
    """Starting a worker using multiproc"""
    process = multiprocessing.Process(target=task, args=args, kwargs=kwargs)
    process.daemon = True
    process.start()
    return process


def main_single(
    IN_HOST, IN_PORT, OUT_HOST, OUT_PORT, OUT_BIND, 
    worker_type, identifier, dataset, model, threshold, verbose
):
    """Runs polling on a single worker"""
    context = SerializingContext()
    if worker_type == 'image':
        worker = ImageObjectDetector
    elif worker_type == 'radar':
        worker = RadarObjectDetector
    else:
        raise NotImplementedError(worker_type)
    detector = worker(
        context,
        IN_HOST,
        IN_PORT,
        OUT_HOST,
        OUT_PORT,
        OUT_BIND,
        identifier=identifier,
        dataset=dataset,
        model=model,
        threshold=threshold,
        verbose=verbose,
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
    main_partial = partial(main_single, args.in_host, args.in_port, args.out_host, args.out_port, args.out_bind)

    # -- start image workers
    for i in range(args.n_image_workers):
        proc = start_worker(
            main_partial,
            worker_type='image',
            identifier=i,
            dataset=args.image_dataset,
            model=args.image_model,
            threshold=args.image_threshold,
            verbose=args.verbose,
        )
        procs.append(proc)

    # -- start radar workers
    for i in range(args.n_radar_workers):
        proc = start_worker(
            main_partial,
            worker_type='radar',
            identifier=i,
            dataset=args.radar_dataset,
            model=args.radar_model,
            threshold=args.radar_threshold,
            verbose=args.verbose,
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

    # -- image model parameters
    parser.add_argument(
        "--n_image_workers",
        type=int,
        default=2,
        help="Number of image detection workers"
    )
    parser.add_argument(
        "--image_model",
        choices=["none", "fasterrcnn"],
        default="none",
        help="Perception model name to run",
    )
    parser.add_argument(
        "--image_dataset",
        choices=['coco-person'],
        default='coco-person',
        help="Perception dataset to base model on"
    )
    parser.add_argument(
        "--image_threshold",
        type=float,
        default=0.5,
        help="Confidence score threshold for image perception"
    )
    # -- radar model parameters
    parser.add_argument(
        "--n_radar_workers",
        type=int,
        default=0,
        help="Number of radar detection workers"
    )
    parser.add_argument(
        "--radar_model",
        choices=["none"],
        default="none",
        help="Radar perception model name to run",
    )
    parser.add_argument(
        "--radar_dataset",
        choices=["none"],
        default="none",
        help="Radar perception dataset to base model on"
    )
    parser.add_argument(
        "--radar_threshold",
        type=float,
        default=0.5,
        help="Confidence score threshold for radar perception"
    )

    # -- host/port information
    parser.add_argument(
        "--in_host", default="localhost", type=str, help="Hostname to connect to"
    )
    parser.add_argument(
        "--in_port", default=5551, type=int, help="Port to connect to server/broker"
    )
    parser.add_argument(
        "--out_host",
        default="localhost",
        type=str,
        help="Hostname to connect output to",
    )
    parser.add_argument(
        "--out_port", default=5553, type=int, help="Port to connect output data to"
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
