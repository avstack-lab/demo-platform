import argparse
import logging

import zmq

import jumpstreet


def pprint(msg, end="\n"):
    print(f"::frontend-dummy::{msg}", end=end, flush=True)


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
        BIND=True,
        subopts=b"",
    )
    frontend_images = jumpstreet.utils.init_some_end(
        cls=None,
        context=context,
        end_type="frontend",
        pattern=zmq.SUB,
        HOST=args.host,
        PORT=args.port_images,
        BIND=True,
        subopts=b"",
    )

    # -- set up polling
    poller = zmq.Poller()
    poller.register(frontend_tracks, zmq.POLLIN)
    poller.register(frontend_images, zmq.POLLIN)

    # -- run loops
    try:
        while True:
            socks = dict(poller.poll())
            # -- get video buffer to send to display
            if frontend_images in socks:
                msg, image = frontend_images.recv_array(copy=False)
                if args.verbose:
                    pprint(f"received image of shape {image.shape}!")

            # -- get trigger from track data
            if frontend_tracks in socks:
                tracks = frontend_tracks.recv()
                if args.verbose:
                    pprint("received tracks!")

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
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    main(args)
