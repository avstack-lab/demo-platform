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

    def __init__(self, display_cam_id, context, HOST, PORT_TRACKS, PORT_IMAGES, dt_delay=0.1):
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
        self.video_buffer = jumpstreet.buffer.BasicDataBuffer(identifier=0, max_size=30)
        self.track_buffer = jumpstreet.buffer.BasicDataBuffer(identifier=0, max_size=30)
        self.muxer = jumpstreet.muxer.VideoTrackMuxer(
            self.video_buffer, self.track_buffer, identifier=0)
        self.t_first_image = None
        self.t_last_image = None
        self.t_first_emit = None
        self.t_last_emit = None
        self.dt_delay = dt_delay

    def run(self):
        from avstack.modules.tracking.tracks import get_data_container_from_line
        from avstack.datastructs import DataContainer
        self.video_buffer.init()
        self.track_buffer.init()
        self.muxer.init()
        # -- put the muxer process on a thread that executes at a certain rate
        self.muxer.start_continuous_process_thread(execute_rate=100, t_max_delta=0.1)
        try:
            while True:
                socks = dict(self.poller.poll(timeout=1))  # timeout in ms

                # -- add video data to buffer
                if self.frontend_images in socks:
                    msg, image = self.frontend_images.recv_array(
                        copy=False
                    )
                    timestamp = msg['timestamp']
                    frame = msg['frame']
                    print(f"received frame at time: {timestamp}")
                    identifier = msg['identifier']
                    image_data_container = DataContainer(frame=frame, timestamp=timestamp,
                                                    data=image, source_identifier=identifier)
                    self.video_buffer.push(image_data_container)

                # -- add track data to buffer
                if self.frontend_tracks in socks:
                    key, data = self.frontend_tracks.recv_multipart()
                    track_data_container = get_data_container_from_line(data.decode(), identifier_override=args.display_cam_id)
                    self.track_buffer.push(track_data_container)
                    print(track_data_container.timestamp)

                # -- emit an image, subject to a delay factor
                emit = False
                t_now = time.time()  # should we redo time.time later on?
                if not self.muxer.empty():
                    t_next_image = self.muxer.top(self.display_cam_id)[0]  # 0 is the identifier for a single camera
                    if self.t_last_emit is None:
                        self.t_last_emit = t_now  # say now is the last emit time
                    if self.t_last_image is None:
                        # first time, put it on a delay
                        self.t_first_image = t_next_image
                        if (t_now - self.t_last_emit) >= self.dt_delay-1e-6:
                            emit = True
                    else:
                        # every other time, match the time difference and maintain the original delay
                        # if we instead tried to match the rate between pairs of images, we could end
                        # up with a gradually increasing global delay factor that would be very bad!
                        dt_from_first_image = t_next_image - self.t_first_image
                        dt_from_first_emit = t_now - self.t_first_emit
                        if dt_from_first_emit >= (dt_from_first_image + self.dt_delay-1e-6):
                            emit = True
                if emit:
                    image_out = self.muxer.pop(self.display_cam_id)  # 0 is the identifier for a single camera
                    t_now_again = time.time()
                    if self.t_first_emit is None:
                        self.t_first_emit = t_now_again
                    self.dt_last = t_now_again - self.t_last_emit  # this should be about 
                    self.t_last_emit = t_now_again
                    self.t_last_image = t_next_image
                    self.update.emit([image_out.data])

        except Exception as e:
            logging.warning(e, exc_info=True)

        finally:
            self.frontend_tracks.close()
            self.frontend_images.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Front end process")
    parser.add_argument("--display_cam_id", type=str, help="Identifier name of camera image to display")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port_tracks", type=int, default=5554)
    parser.add_argument("--port_images", type=int, default=5552)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    context = jumpstreet.context.SerializingContext()
    main_loop = MainLoop(display_cam_id=args.display_cam_id, context=context, HOST=args.host, PORT_TRACKS=args.port_tracks,
        PORT_IMAGES=args.port_images)

    app = QApplication(sys.argv)
    display = jumpstreet.display.StreamThrough(main_loop=main_loop, width=args.width, height=args.height, identifier=0, verbose=args.verbose)
    display.start()
    sys.exit(app.exec())