"""
Runs the front-end which includes:
- video buffer
- trigger
- display
"""

import argparse
import logging
import zmq
import time
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject, pyqtSignal
import jumpstreet


class MainLoop(QObject):
    update = pyqtSignal(object)

    def __init__(self, context, HOST, PORT_TRACKS, PORT_IMAGES, dt_delay=0.10):
        super().__init__()
        self.quit_flag = False

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
        self.video_buffer = jumpstreet.buffer.BasicDataBuffer(identifier=0, max_size=30)
        self.track_buffer = jumpstreet.buffer.BasicDataBuffer(identifier=0, max_size=30)
        self.muxer = jumpstreet.muxer.VideoTrackMuxer(
            self.video_buffer, self.track_buffer, identifier=0)
        self.t_last_image = None
        self.t_last_emit = None
        self.dt_delay = dt_delay

    def run(self):
        from avstack.modules.tracking.tracks import get_data_container_from_line
        from avstack.datastructs import DataContainer
        self.video_buffer.init()
        self.track_buffer.init()
        self.muxer.init()
        try:
            while True:
                socks = dict(self.poller.poll(timeout=10))  # timeout in ms
                # TODO: have a REALLY short limit on the poller so we don't delay emitting frames

                # -- add video data to buffer
                if self.frontend_images in socks:
                    msg, image = self.frontend_images.recv_array(
                        copy=False
                    )
                    timestamp = msg['timestamp']
                    frame = msg['frame']
                    identifier = msg['identifier']
                    image_data_container = DataContainer(frame=frame, timestamp=timestamp,
                                                    data=image, source_identifier=identifier)
                    self.video_buffer.push(image_data_container)

                # -- add track data to buffer
                if self.frontend_tracks in socks:
                    key, data = self.frontend_tracks.recv_multipart()
                    track_data_container = get_data_container_from_line(data.decode(), identifier_override=0)
                    self.track_buffer.push(track_data_container)

                # -- run the muxer and load an image when we're ready
                # TODO: put the processor on a separate thread???
                self.muxer.process()

                # -- emit an image steadily, subject to a delay factor
                emit = False
                t_now = time.time()  # should we redo time.time later on?
                if not self.muxer.empty():
                    t_next_image = self.muxer.top(0)[0]  # 0 is the identifier for a single camera
                    if self.t_last_emit is None:
                        self.t_last_emit = t_now  # say now is the last emit time
                    if self.t_last_image is None:
                        if (t_now - self.t_last_emit) > self.dt_delay:
                            emit = True
                    else:
                        if (t_now - self.t_last_emit) > (t_next_image - self.t_last_image + self.dt_delay):
                            emit = True
                if emit:
                    image_out = self.muxer.pop(0)  # 0 is the identifier for a single camera
                    self.update.emit([image_out.data])
                    self.t_last_emit = t_now
                    self.t_last_image = t_now

        except Exception as e:
            logging.warning(e, exc_info=True)

        finally:
            self.frontend_tracks.close()
            self.frontend_images.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Front end process")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port_tracks", type=int, default=5554)
    parser.add_argument("--port_images", type=int, default=5552)

    args = parser.parse_args()

    context = jumpstreet.context.SerializingContext()
    main_loop = MainLoop(context=context, HOST=args.host, PORT_TRACKS=args.port_tracks,
        PORT_IMAGES=args.port_images)

    app = QApplication(sys.argv)
    display = jumpstreet.display.StreamThrough(main_loop=main_loop, identifier=0)
    display.start()
    sys.exit(app.exec())