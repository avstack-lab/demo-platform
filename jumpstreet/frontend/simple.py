"""
Runs the front-end which includes:
- video buffer
- trigger
- display
"""

import argparse
import logging
import sys

import numpy as np
import zmq
from avstack.datastructs import BasicDataBuffer, DataContainer
from avstack.modules.tracking.tracks import get_data_container_from_line
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
        display_cam_id,
        context,
        frontend_images,
        frontend_tracks,
        dt_delay=0.1,
        verbose=False,
        debug=False,
    ):
        super().__init__()
        self.quit_flag = False
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
            HOST=frontend_images.host,
            PORT=frontend_images.port,
            BIND=frontend_images.bind,
            subopts=b"",
        )
        self.dt_delay = dt_delay

        # -- set up polling
        self.poller = zmq.Poller()
        self.poller.register(self.frontend_tracks, zmq.POLLIN)
        self.poller.register(self.frontend_images, zmq.POLLIN)

        # -- set up processes
        self.trigger = jumpstreet.trigger.AlwaysTrigger(identifier=0)
        self.video_buffer = BasicDataBuffer(max_size=30)
        self.track_buffer = BasicDataBuffer(max_size=30)
        self.muxer = jumpstreet.muxer.VideoTrackMuxer(
            self.video_buffer,
            self.track_buffer,
            identifier=0,
            dt_delay=self.dt_delay,
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
                    msg, image = self.frontend_images.recv_array(copy=False)

                    # -- decompress data (NZ)
                    decoded_frame = imdecode(image, IMREAD_COLOR)
                    image = np.array(decoded_frame)  # ndarray with d = (h, w, 3)

                    timestamp = msg["timestamp"]
                    frame = msg["frame"]
                    if self.debug:
                        self.print(f"received frame at time: {timestamp}", end="\n")
                    identifier = msg["identifier"]
                    image_data_container = DataContainer(
                        frame=frame,
                        timestamp=timestamp,
                        data=image,
                        source_identifier=identifier,
                    )
                    self.video_buffer.push(image_data_container)

                    # -- update display window dimensions to fit image
                    # h, w, _ = image.shape
                    # padding = (
                    #     1.1  # accounts for scroll area and padding in Image Viewer
                    # )
                    # self.display.resize(int(padding * w), int(padding * h))

                # -- add track data to buffer
                if self.frontend_tracks in socks:
                    key, data = self.frontend_tracks.recv_multipart()
                    track_data_container = get_data_container_from_line(
                        data.decode(),
                        identifier_override=self.display_cam_id,
                    )
                    if self.debug:
                        self.print(
                            f"received tracks at time: {track_data_container.timestamp}",
                            end="\n",
                        )
                    self.track_buffer.push(track_data_container)

                # -- emit an image, subject to a delay factor
                image_out = self.muxer.emit_one()
                if (self.display_cam_id in image_out) and len(
                    image_out[self.display_cam_id]
                ) > 0:
                    if self.debug:
                        self.print(f"emitting image to display", end="\n")
                    self.update.emit([image_out[self.display_cam_id].data])

        except Exception as e:
            logging.warning(e, exc_info=True)

        finally:
            self.frontend_tracks.close()
            self.frontend_images.close()


def main(config):
    context = jumpstreet.context.SerializingContext(config.display.io_threads)
    main_loop = MainLoop(
        display_cam_id=config.display.camera_id,
        context=context,
        frontend_images=config.frontend_images,
        frontend_tracks=config.frontend_tracks,
        verbose=config.verbose,
        debug=config.debug,
    )

    app = QApplication(sys.argv)
    display = jumpstreet.display.StreamThrough(
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
