"""
Runs the front-end which includes:
- video buffer
- trigger
- display
"""

import argparse
import json
import logging
import sys

import numpy as np
import zmq
from avstack.calibration import CalibrationDecoder
from avstack.datastructs import BasicDataBuffer, DelayManagedDataBuffer
from avstack.modules.tracking.tracks import TrackContainerDecoder
from avstack.sensors import ImageData
from cv2 import IMREAD_COLOR, imdecode
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication

import jumpstreet
from jumpstreet.utils import config_as_namespace


class MainLoop(QObject):
    update = pyqtSignal(object)
    NAME = "frontend"

    def __init__(
        self,
        config,
        display_cam_id,
        context,
        frontend_images,
        frontend_tracks,
        verbose=False,
        debug=False,
    ):
        super().__init__()
        self.quit_flag = False
        self.config = config
        self.display_cam_id = display_cam_id
        self.verbose = verbose
        self.debug = debug
        self.identifier = 0

        # -- set up frontend data receivers
        self.frontend_tracks = jumpstreet.utils.init_some_end(
            cls=None,
            context=context,
            end_type="frontend",
            pattern=zmq.SUB,
            TRANSPORT=frontend_tracks.transport,
            HOST=frontend_tracks.host,
            PORT=frontend_tracks.port,
            BIND=frontend_tracks.bind,
            subopts=b"",
        )
        self.frontend_images = jumpstreet.utils.init_some_end(
            cls=None,
            context=context,
            end_type="frontend",
            pattern=zmq.SUB,
            TRANSPORT=frontend_images.transport,
            HOST=frontend_images.host,
            PORT=frontend_images.port,
            BIND=frontend_images.bind,
            subopts=b"",
        )

        # -- set up polling
        self.poller = zmq.Poller()
        self.poller.register(self.frontend_tracks, zmq.POLLIN)
        self.poller.register(self.frontend_images, zmq.POLLIN)

        # -- set up processes
        self.trigger = jumpstreet.trigger.AlwaysTrigger(identifier=0)
        self.video_buffer = DelayManagedDataBuffer(
            dt_delay=self.config.display.dt_delay, max_size=100, method="real-time"
        )
        self.track_buffer = BasicDataBuffer(max_size=30)
        self.muxer = jumpstreet.muxer.VideoTrackMuxer(
            self.video_buffer,
            self.track_buffer,
            identifier=0,
            verbose=self.verbose,
            debug=self.debug,
        )

    def print(self, msg, end="", flush=True):
        try:
            name = self.NAME
        except AttributeError as e:
            name = self.name
        print(f"::{name}-{self.identifier}::{msg}", end=end, flush=flush)

    def run(self):
        # -- put the muxer process on a thread that executes at a certain rate
        self.muxer.start_continuous_process_thread(execute_rate=100, t_max_delta=0.1)
        try:
            while True:
                socks = dict(self.poller.poll(timeout=1))  # timeout in ms

                # -- add video data to buffer
                if self.frontend_images in socks:
                    msg, array = self.frontend_images.recv_array(copy=False)

                    # -- decompress data
                    if msg["encoded"]:
                        decoded_frame = imdecode(array, IMREAD_COLOR)
                        array = np.array(decoded_frame)  # ndarray with d = (h, w, 3)
                    timestamp = msg["timestamp"]
                    frame = msg["frame"]
                    identifier = msg["identifier"]
                    calib = json.loads(msg["calibration"], cls=CalibrationDecoder)
                    image = ImageData(
                        timestamp=timestamp,
                        frame=frame,
                        source_ID=identifier,
                        source_name="camera",
                        data=array,
                        calibration=calib,
                    )
                    image.source_identifier = self.display_cam_id  # HACK
                    if self.debug:
                        self.print(f"received frame at time: {timestamp}", end="\n")
                    self.video_buffer.push(image)

                # -- add track data to buffer
                if self.frontend_tracks in socks:
                    key, data = self.frontend_tracks.recv_multipart()
                    track_data_container = json.loads(
                        data.decode(), cls=TrackContainerDecoder
                    )
                    # HACK: overrite the camera identifier
                    track_data_container.source_identifier = self.display_cam_id
                    # identifier_override=self.display_cam_id,

                    if self.debug:
                        self.print(
                            f"received tracks at time: {track_data_container.timestamp}",
                            end="\n",
                        )
                    self.track_buffer.push(track_data_container)

                # -- emit an image, subject to a delay factor
                image_out = self.muxer.emit_one()
                if self.display_cam_id in image_out:
                    if self.debug:
                        self.print(f"emitting image to display", end="\n")
                    try:
                        self.update.emit([image_out[self.display_cam_id].data])
                    except KeyboardInterrupt:
                        pass  # weirdness...

        except Exception as e:
            logging.warning(e, exc_info=True)

        finally:
            self.frontend_tracks.close()
            self.frontend_images.close()


def main(config):
    context = jumpstreet.context.SerializingContext(config.display.io_threads)
    main_loop = MainLoop(
        config=config,
        display_cam_id=config.display.camera_id,
        context=context,
        frontend_images=config.frontend_images,
        frontend_tracks=config.frontend_tracks,
        verbose=config.verbose,
        debug=config.debug,
    )

    app = QApplication(sys.argv)
    display = jumpstreet.display.base.StreamThrough(
        main_loop=main_loop,
        width=config.display.width / 2,
        height=config.display.height / 2,
        identifier=0,
        verbose=config.verbose,
        debug=config.debug,
    )
    display.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Initialize a frontend display")
    parser.add_argument("--config", default="frontend/default.yml")
    args = parser.parse_args()
    config = config_as_namespace(args.config)
    main(config)
