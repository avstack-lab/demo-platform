#!/usr/bin/env python3

import argparse
import logging
import multiprocessing
from functools import partial
from time import sleep

import cv2
import numpy as np
import zmq
from avstack.calibration import read_calibration_from_line
from avstack.modules.perception.detections import (
    format_data_container_as_string,
)
from avstack.modules.perception.object2dfv import MMDetObjectDetector2D
from avstack.sensors import ImageData
from avstack.modules.perception.detections import RazDetection
from avstack.datastructs import DataContainer

from jumpstreet.context import SerializingContext
from jumpstreet.utils import BaseClass, config_as_namespace, init_some_end


class ObjectDetector(BaseClass):
    def __init__(
        self,
        context,
        frontend,
        backend,
        identifier,
        detector,
        dataset,
        model,
        threshold,
        verbose=False,
        debug=False,
    ) -> None:
        """Set up front and back ends

        Front end: req
        Back end: pub
        """
        super().__init__(
            name=self.NAME, identifier=identifier, verbose=verbose, debug=debug
        )
        self.frontend = init_some_end(
            self,
            context,
            "frontend",
            zmq.REQ,
            frontend.transport,
            frontend.host,
            frontend.port,
            BIND=frontend.bind,
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
        self.set_model(detector, dataset, model, threshold)
        if self.verbose:
            self.print("initialized perception model!", end="\n")

        # -- ready to go (need this!)
        self.frontend.send(f"READY-{self.WORKER}".encode())
        if self.verbose:
            self.print(f"ready to start", end="\n")

    def poll(self):
        """Poll for messages

        Address is the place to send back data
        """
        # -- get data from frontend
        address, metadata, array = self.frontend.recv_array_multipart(copy=True)

        # -- decompress data (NZ)
        if "camera" in metadata["msg"]["identifier"]:
            decoded_frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
            array = np.array(decoded_frame)  # ndarray with d = (h, w, 3)
            metadata["shape"] = array.shape
        elif "radar" in metadata["msg"]["identifier"]:
            pass
        else:
            raise NotImplementedError(metadata["identifier"])

        # -- process data
        detections = self.detect(metadata, array)
        if detections is not None:
            detections = format_data_container_as_string(detections).encode()
        else:
            detections = b"No detections yet"

        # -- send data at backend
        self.backend.send_multipart([b"detections", detections])
        # -- say we're ready for more
        self.frontend.send_multipart([address, b"", f"OK-{self.WORKER}".encode()])

    def set_model(self):
        raise NotImplementedError

    def detect(self):
        raise NotImplementedError


class ImageObjectDetector(ObjectDetector):
    NAME = "image-detector"
    WORKER = "camera"

    def __init__(
        self,
        context,
        frontend,
        backend,
        identifier,
        detector="mmdetection",
        dataset="coco-person",
        model="fasterrcnn",
        threshold=0.5,
        verbose=False,
        debug=False,
    ) -> None:
        super().__init__(
            context,
            frontend,
            backend,
            identifier,
            detector,
            dataset,
            model,
            threshold,
            verbose,
            debug,
        )

    def set_model(self, detector, dataset, model, threshold):
        # -- set up perception model
        if detector == "mmdetection":
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
            identifier = metadata["msg"]["identifier"]
            calib = read_calibration_from_line(metadata["msg"]["calibration"])
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
            if self.debug:
                self.print(
                    f"Image frame: {frame:4d}, timestamp: {timestamp:.4f}, {len(detections)} detections", end="\n"
                )
        else:
            detections = None
        return detections


class RadarObjectDetector(ObjectDetector):
    NAME = "radar-detector"
    WORKER = "radar"

    def __init__(
        self,
        context,
        frontend,
        backend,
        identifier,
        detector=None,
        dataset=None,
        model="passthrough",
        threshold=0.5,
        verbose=True,
        debug=False,
    ) -> None:
        super().__init__(
            context,
            frontend,
            backend,
            identifier,
            detector,
            dataset,
            model,
            threshold,
            verbose,
            debug,
        )
        self.passthrough = model == "passthrough"

    def set_model(self, detector, dataset, model, threshold):
        if model == "passthrough":
            def msmts_to_dc(metadata, array):
                dets = []
                for i in range(array.shape[0]):
                    dets.append(RazDetection(source_identifier=metadata["msg"]["identifier"], raz=array[i,:2]))
                return DataContainer(frame=metadata["msg"]["frame"], timestamp=metadata["msg"]["timestamp"], data=dets, source_identifier=metadata["msg"]["identifier"])
            self.model = msmts_to_dc
        else:
            raise NotImplementedError(model)

    def detect(self, metadata, array):
        array = np.reshape(array, metadata["shape"])
        detections = self.model(metadata, array)
        return detections


def start_worker(task, *args, **kwargs):
    """Starting a worker using multiproc"""
    process = multiprocessing.Process(target=task, args=args, kwargs=kwargs)
    process.daemon = True
    process.start()
    return process


def main_single(
    frontend,
    backend,
    worker_type,
    identifier,
    detector,
    dataset,
    model,
    threshold,
    verbose,
    debug,
):
    """Runs polling on a single worker"""
    context = SerializingContext()
    if worker_type == "image":
        worker = ImageObjectDetector
    elif worker_type == "radar":
        worker = RadarObjectDetector
    else:
        raise NotImplementedError(worker_type)
    detector = worker(
        context,
        frontend,
        backend,
        identifier=identifier,
        detector=detector,
        dataset=dataset,
        model=model,
        threshold=threshold,
        verbose=verbose,
        debug=debug,
    )
    try:
        while True:
            detector.poll()
    except Exception as e:
        logging.warning(e, exc_info=True)
    finally:
        detector.close()


def main(config):
    """Run object detection workers"""
    procs = []
    main_partial = partial(
        main_single,
        config.frontend,
        config.backend,
    )

    # -- start image workers
    for i in range(config.workers.image.n_workers):
        proc = start_worker(
            main_partial,
            worker_type="image",
            identifier=i,
            detector=config.workers.image.detector,
            dataset=config.workers.image.dataset,
            model=config.workers.image.model,
            threshold=config.workers.image.threshold,
            verbose=config.verbose,
            debug=config.debug,
        )
        procs.append(proc)

    # -- start radar workers
    for i in range(config.workers.radar.n_workers):
        proc = start_worker(
            main_partial,
            worker_type="radar",
            identifier=i,
            detector=config.workers.radar.detector,
            dataset=config.workers.radar.dataset,
            model=config.workers.radar.model,
            threshold=config.workers.radar.threshold,
            verbose=config.verbose,
            debug=config.debug,
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
    parser.add_argument("--config", default="detection/default.yml")
    args = parser.parse_args()
    config = config_as_namespace(args.config)
    main(config)
