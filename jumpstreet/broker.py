#!/usr/bin/env python3

from __future__ import print_function

import argparse
import logging

import zmq

from jumpstreet.context import SerializingContext
from jumpstreet.utils import BaseClass, config_as_namespace, init_some_end


class LoadBalancingBrokerXSub(BaseClass):
    """Load Balancing Broker

    A load-balancing broker uses a router for both frontend and backend
    """

    NAME = "lb-broker-xsub"

    def __init__(
        self,
        context,
        frontend,
        backend,
        backend_other,
        identifier=0,
        verbose=False,
        debug=False,
    ) -> None:
        super().__init__(self.NAME, identifier, verbose=verbose, debug=debug)
        self.frontend = init_some_end(
            self,
            context,
            "frontend",
            zmq.XSUB,
            frontend.transport,
            frontend.host,
            frontend.port,
            BIND=frontend.bind,
            HWM=frontend.highwatermark,
        )
        self.backend = init_some_end(
            self,
            context,
            "backend",
            zmq.ROUTER,
            backend.transport,
            backend.host,
            backend.port,
            BIND=backend.bind,
        )
        self.backend_xpub = init_some_end(
            self,
            context,
            "backend-xpub",
            zmq.XPUB,
            backend_other.transport,
            backend_other.host,
            backend_other.port,
            BIND=backend_other.bind,
        )
        self.backend_ready = {"camera": False, "radar": False}
        self.workers = {"camera": [], "radar": []}
        self.print(f"initializing poller...", end="")
        self.poller = zmq.Poller()
        self.poller.register(self.backend, zmq.POLLIN)
        self.poller.register(self.frontend, zmq.POLLIN)
        self.poller.register(self.backend_xpub, zmq.POLLIN)
        self.print("done", end="\n")

    def poll(self):
        socks = dict(self.poller.poll(timeout=10))

        # --- handle worker activity on the backend
        if self.backend in socks:
            request = self.backend.recv_multipart()
            worker, empty, client = request[:3]  # TODO: add worker_type
            worker_type = client.decode().split("-")[1]
            self.workers[worker_type].append(worker)
            for k in self.workers:
                if self.workers[k] and not self.backend_ready[k]:
                    self.backend_ready[k] = True

        # --- handle incoming data on the frontend
        if self.frontend in socks and socks[self.frontend] == zmq.POLLIN:
            msg, array = self.frontend.recv_array(copy=False)
            if array is not None:
                # --- handle different data types
                pass_data = False
                if "camera" in msg["identifier"]:
                    data_type = "camera"
                    pass_data = True
                elif "radar" in msg["identifier"]:
                    data_type = "radar"
                    pass_data = True

                # -- pass on the data
                if pass_data:
                    # -- primary worker
                    if self.debug:
                        self.print(
                            f"received {data_type} array of size {array.shape}", end="\n"
                        )
                    if self.backend_ready[data_type]:
                        worker = self.workers[data_type].pop(0)
                        client = f"OK-{data_type}".encode()  # TODO: why is this needed???
                        self.backend.send_array_envelope(
                            worker, client, msg, array, copy=False
                        )
                    else:
                        if self.debug:
                            self.print(
                                f"broker had {data_type} data to send but no worker ready",
                                end="\n",
                            )

                    # -- secondary xpub (for display only)
                    if data_type == "camera":
                        self.backend_xpub.send_array(array, msg=msg, copy=False)

                # -- check workers
                for k in self.workers:
                    if not self.workers[k]:
                        self.backend_ready[k] = False

        # --- handle subscription requests
        if self.backend_xpub in socks and socks[self.backend_xpub] == zmq.POLLIN:
            msg = self.backend_xpub.recv_multipart()
            if self.verbose:
                self.print(
                    f'received subscription message: "{msg[0].decode("utf-8")}"',
                    end="\n",
                )
            self.frontend.send_multipart(msg)

    def close(self):
        super().close()
        self.backend_xpub.close()


def init_broker(broker):
    if broker.type.lower() == "lb":
        raise NotImplementedError
    elif broker.type.lower() == "lb_with_xsub_extra_xpub":
        return LoadBalancingBrokerXSub
    else:
        raise NotImplementedError(broker.type)


def main(config):
    context = SerializingContext(config.broker.io_threads)
    broker = init_broker(config.broker)(
        context,
        frontend=config.frontend,
        backend=config.backend,
        verbose=config.verbose,
        debug=config.debug,
        backend_other=config.backend_other,
    )
    try:
        while True:
            broker.poll()
    except Exception as e:
        logging.warning(e, exc_info=True)
    finally:
        broker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Initialize a broker")
    parser.add_argument("--config", default="broker/default.yml")
    args = parser.parse_args()
    config = config_as_namespace(args.config)
    main(config)
