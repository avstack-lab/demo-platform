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


def main(args):
    context = jumpstreet.context.SerializingContext()

    # -- set up frontend data receivers
    frontend_tracks = jumpstreet.utils.init_some_end(
        cls=None,
        context=context,
        end_type="frontend",
        pattern=zmq.SUB,
        HOST=args.host,
        PORT=args.port_tracks,
        BIND=False,
        subopts="tracks",
    )
    frontend_images = jumpstreet.utils.init_some_end(
        cls=None,
        context=context,
        end_type="frontend",
        pattern=zmq.SUB,
        HOST=args.host,
        PORT=args.port_images,
        BIND=False,
        subopts="images",
    )

    # -- set up polling
    poller = zmq.Poller()
    poller.register(frontend_tracks, zmq.POLLIN)
    poller.register(frontend_images, zmq.POLLIN)

    # -- set up processes
    trigger = jumpstreet.trigger.AlwaysTrigger(identifier=0)
    buffer = jumpstreet.buffer.VideoBuffer(identifier=0)
    display = jumpstreet.display.ConfirmationDisplay(identifier=0)
    display.start()

    # -- run loops
    try:
        while True:
            socks = dict(poller.poll())
            # -- get video buffer to send to display
            if frontend_images in socks:
                address, metadata, image = frontend_images.recv_array_multipart(
                    copy=False
                )
                buffer.store(
                    t=metadata["timestamp"], cam_id=metadata["camera_ID"], image=image
                )

            # -- get trigger from track data
            if frontend_tracks in socks:
                tracks = frontend_tracks.recv()
                t_start, t_end = trigger(tracks)
                images_out = buffer.trigger(t_start, t_end)
                if images_out is not None:
                    display(images_out)

    except Exception as e:
        logging.warning(e, exc_info=True)

    finally:
        frontend_tracks.close()
        frontend_images.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Front end process")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port_tracks", type=int, default=5554)
    parser.add_argument("--port_images", type=int, default=5552)

    args = parser.parse_args()
    main(args)
