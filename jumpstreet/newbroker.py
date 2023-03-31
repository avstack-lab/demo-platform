#!/usr/bin/env python3

from __future__ import print_function

import argparse
import logging

import numpy as np
import zmq

# from jumpstreet.context import SerializingContext
from utils import BaseClass, init_some_end


class SensorDataReceiver(BaseClass):
    """Load Balancing Broker

    A load-balancing broker uses a router for both frontend and backend
    """

    NAME = "receiver"

    def __init__(
        self,
        context,
        FRONTEND=6550,
        BACKEND=6551,
        BACKEND_OTHER=6552,
        identifier=0,
        verbose=False,
    ) -> None:
        super().__init__(self.NAME, identifier)
        self.verbose = verbose
        self.frontend = init_some_end(
            self, context, "frontend", zmq.SUB, "*", FRONTEND, BIND=True
        )
        self.backend = init_some_end(
            self, context, "backend", zmq.ROUTER, "*", BACKEND, BIND=True
        )
        self.backend_ready = False
        self.workers = []
        self.print(f"initializing poller...", end="")
        self.poller = zmq.Poller()
        self.poller.register(self.backend, zmq.POLLIN)
        self.poller.register(self.frontend, zmq.POLLIN)
        print("done")

    def poll(self):
        socks = dict(self.poller.poll())

        # --- handle worker activity on the backend
        if self.backend in socks:
            request = self.backend.recv_multipart()
            worker, empty, client = request[:3]
            self.workers.append(worker)
            if self.workers and not self.backend_ready:
                self.backend_ready = True

        # --- handle incoming data on the frontend
        if self.frontend in socks and socks[self.frontend] == zmq.POLLIN:
            msg, array = self.frontend.recv_array(copy=False)
            # print(f'received image of size {array.shape}')

            #  -- Route data to last-used worker, if ready
            if self.backend_ready:
                worker = self.workers.pop(0)
                client = b"N/A"
                self.backend.send_array_envelope(worker, client, msg, array, copy=False)
                if not self.workers:
                    self.backend_ready = False
                # else:
                #     print(worker, msg)
                #     raise RuntimeError("No worker available")

            # -- handle secondary xpub
            self.backend_xpub.send_array(array, msg=msg, copy=False)

        # --- handle subscription requests
        if self.backend_xpub in socks and socks[self.backend_xpub] == zmq.POLLIN:
            msg = self.backend_xpub.recv_multipart()
            # print(f'received subscription message: "{msg[0].decode("utf-8")}"')
            self.frontend.send_multipart(msg)

    def close(self):
        super().close()
        self.backend_xpub.close()


# NZ updated for receiver.py
def init_broker(broker_type):
    if broker_type.lower() == "receiver":
        return SensorDataReceiver
    else:
        raise NotImplementedError(broker_type)


def main(args):
    context = SerializingContext()
    broker = init_broker(args.broker)(
        context,
        FRONTEND=args.frontend,
        BACKEND=args.backend,
        verbose=args.verbose,
        BACKEND_OTHER=args.backend_other,
    )
    try:
        while True:
            broker.poll()
    except Exception as e:
        logging.warning(e, exc_info=True)
    finally:
        broker.close()


if __name__ == "__main__":
    print(f"Running receiver.py as {__name__}")
    # parser = argparse.ArgumentParser("Initialize a broker")
    # parser.add_argument(
    #     "broker", choices=["lb", "lb_with_xsub_extra_xpub"],
    #     type=str, help="Selection of broker type"
    # )
    # parser.add_argument(
    #     "--frontend", type=int, default=5550, help="Frontend port number (clients)"
    # )
    # parser.add_argument(
    #     "--backend", type=int, default=5551, help="Backend port number (workers)"
    # )
    # parser.add_argument(
    #     "--backend_other", type=int, help="Extra backend port (used only in select classes)"
    # )
    # parser.add_argument("--verbose", action="store_true")
    # args = parser.parse_args()
    # main(args)
