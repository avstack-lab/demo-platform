"""
Runs the front-end which includes:
- video buffer
- trigger
- display
"""

import argparse
import logging
import zmq
import jumpstreet
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject, pyqtSignal


class MainLoop(QObject):
    update = pyqtSignal(object)

    def __init__(self, context, HOST, PORT_TRACKS, PORT_IMAGES):
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
        # self.buffer = jumpstreet.buffer.VideoBuffer(identifier=0)
        
    def run(self):
        from avstack.modules.tracking.tracks import get_data_container_from_line
        try:
            while True:
                socks = dict(self.poller.poll())

                # -- get video buffer to send to display
                if self.frontend_images in socks:
                    msg, image = self.frontend_images.recv_array(
                        copy=False
                    )
                    timestamp = msg['timestamp']
                    frame = msg['frame']
                    identifier = msg['identifier']
                    # self.buffer.store(
                    #     t=t, cam_id=cam_id, image=image
                    # )
                    print('sending image to display')
                    self.update.emit([image])

                # -- get trigger from track data
                if self.frontend_tracks in socks:
                    key, data = self.frontend_tracks.recv_multipart()
                    tracks = get_data_container_from_line(data.decode())
                    print(tracks)
                    # t_start, t_end = self.trigger(tracks)
                    # images_out = self.buffer.trigger(t_start, t_end)
                    # self.image_buffer.emit(images_out)

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