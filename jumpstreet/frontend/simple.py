"""
Runs the front-end which includes:
- video buffer
- trigger
- display
"""

import argparse
import logging
import sys
import time

import numpy as np
import zmq
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication

import jumpstreet


class MainLoop(QObject):
    update = pyqtSignal(object)

    def __init__(
        self, display_cam_id, context, HOST, PORT_TRACKS, PORT_IMAGES, dt_delay=0.1
    ):
        super().__init__()
        self.quit_flag = False
        self.display_cam_id = display_cam_id

        # -- set up frontend data receivers
        self.frontend_tracks = jumpstreet.utils.init_some_end(
            cls=None,
            context=context,
            end_type="frontend",
            pattern=zmq.SUB,
            HOST=HOST,
            PORT=PORT_TRACKS,
            BIND=False,
            subopts=b"",
        )
        self.frontend_images = jumpstreet.utils.init_some_end(
            cls=None,
            context=context,
            end_type="frontend",
            pattern=zmq.SUB,
            HOST=HOST,
            PORT=PORT_IMAGES,
            BIND=False,
            subopts=b"",
        )

        # -- set up polling
        self.poller = zmq.Poller()
        self.poller.register(self.frontend_tracks, zmq.POLLIN)
        self.poller.register(self.frontend_images, zmq.POLLIN)

        # -- set up processes
        self.trigger = jumpstreet.trigger.AlwaysTrigger(identifier=0)
        self.video_buffer = None
        self.track_buffer = None
        self.muxer = None
        self.dt_delay = dt_delay

    def run(self):
        # -- need to defer import and init due to Qt error
        from avstack.datastructs import BasicDataBuffer, DataContainer
        from avstack.modules.tracking.tracks import get_data_container_from_line
        from cv2 import IMREAD_COLOR, imdecode

        self.video_buffer = BasicDataBuffer(max_size=30)
        self.track_buffer = BasicDataBuffer(max_size=30)
        self.muxer = jumpstreet.muxer.VideoTrackMuxer(
            self.video_buffer, self.track_buffer, identifier=0, dt_delay=self.dt_delay
        )

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
                    # print(f"received frame at time: {timestamp}")
                    identifier = msg["identifier"]
                    image_data_container = DataContainer(
                        frame=frame,
                        timestamp=timestamp,
                        data=image,
                        source_identifier=identifier,
                    )
                    self.video_buffer.push(image_data_container)

                    # -- update display window dimensions to fit image
                    h, w, _ = image.shape
                    padding = (
                        1.1  # accounts for scroll area and padding in Image Viewer
                    )
                    display.resize(int(padding * w), int(padding * h))

                # -- add track data to buffer
                if self.frontend_tracks in socks:
                    key, data = self.frontend_tracks.recv_multipart()
                    track_data_container = get_data_container_from_line(
                        data.decode(), identifier_override=args.display_cam_id
                    )
                    self.track_buffer.push(track_data_container)

                # -- emit an image, subject to a delay factor
                image_out = self.muxer.emit_one()
                if (self.display_cam_id in image_out) and len(
                    image_out[self.display_cam_id]
                ) > 0:
                    self.update.emit([image_out[self.display_cam_id].data])

        except Exception as e:
            logging.warning(e, exc_info=True)

        finally:
            self.frontend_tracks.close()
            self.frontend_images.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Front end process")
    parser.add_argument(
        "--display_cam_id", type=str, help="Identifier name of camera image to display"
    )
    parser.add_argument("--io_threads", type=int, default=2)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port_tracks", type=int, default=5554)
    parser.add_argument("--port_images", type=int, default=5552)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    context = jumpstreet.context.SerializingContext(args.io_threads)
    main_loop = MainLoop(
        display_cam_id=args.display_cam_id,
        context=context,
        HOST=args.host,
        PORT_TRACKS=args.port_tracks,
        PORT_IMAGES=args.port_images,
    )

    app = QApplication(sys.argv)
    display = jumpstreet.display.StreamThrough(
        main_loop=main_loop,
        width=args.width / 2,
        height=args.height / 2,
        identifier=0,
        verbose=args.verbose,
    )
    display.start()
    sys.exit(app.exec())
