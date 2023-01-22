#!/usr/bin/env python3

from __future__ import print_function
import argparse
import zmq
import logging
import numpy as np
from jumpstreet.utils import init_some_end
from jumpstreet.context import SerializingContext


class LoadBalancingBroker():
    """Load Balancing Broker
    
    A load-balancing broker uses a router for both frontend and backend
    """
    NAME = 'load-balancing-broker'
    
    def __init__(self, context, FRONTEND=5555, BACKEND=5556, verbose=False) -> None:
        self.verbose = verbose        
        self.frontend = init_some_end(self, context, 'frontend', zmq.ROUTER, '*', FRONTEND, BIND=True)
        self.backend = init_some_end(self, context, 'backend', zmq.ROUTER, '*', BACKEND, BIND=True)
        self.backend_ready=False
        self.workers = []
        self.print(f'initializing poller...', end='')
        self.poller = zmq.Poller()
        self.poller.register(self.backend, zmq.POLLIN) 
        self.poller.register(self.frontend, zmq.POLLIN)
        print('done')
        
    @classmethod
    def print(cls, msg, end=''):
        print(f"::{cls.NAME}::{msg}", end=end, flush=True)

    def poll(self):
        sockets = dict(self.poller.poll())

        # --- handle worker activity on the backend
        if self.backend in sockets:
            request = self.backend.recv_multipart()
            worker, empty, client = request[:3]
            self.workers.append(worker)
            if self.workers and not self.backend_ready:
                # Poll for clients now that a worker is available and backend was not ready
                self.poller.register(self.frontend, zmq.POLLIN)
                self.backend_ready = True
            if client != b"READY" and len(request) > 3:
                # If client reply, send rest back to frontend
                empty, reply = request[3:]
                self.frontend.send_multipart([client, b"", reply])

        # --- handle client requests on the frontend
        if self.frontend in sockets:
            # Get next client request, route to last-used worker
            client, metadata, array = self.frontend.recv_array_envelope()
            if metadata["msg"] == "READY":
                # -- client discovery and acknowledgement
                reply = b"OK"
                self.frontend.send_multipart([client, b"", reply])
                # self.poller.unregister(self.frontend)
            elif self.backend_ready and ("IMAGE" in metadata["msg"]):
                # -- client requests
                worker = self.workers.pop(0)
                self.backend.send_array_envelope(worker, client, array, metadata["msg"])
                if not self.workers:
                    # Don't poll clients if no workers are available and set backend_ready flag to false
                    self.poller.unregister(self.frontend)
                    self.backend_ready = False
            else:
                raise NotImplemented(metadata["msg"])

    def close(self):
        self.frontend.close()
        self.backend.close()


def init_broker(broker_type):
    if broker_type.lower() == 'loadbalancing':
        return LoadBalancingBroker
    else:
        raise NotImplementedError(broker_type)


def main(args):
    # context = zmq.Context.instance()
    context = SerializingContext()
    broker = init_broker(args.broker)(context,
        FRONTEND=args.frontend, BACKEND=args.backend, verbose=args.verbose)
    try:
        while True:
            broker.poll()
    except Exception as e:
        logging.warning(e, exc_info=True)
    finally:
        broker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Initialize a broker')
    parser.add_argument('broker', choices=['loadbalancing'], type=str, help='Selection of broker type')
    parser.add_argument('--frontend', type=int, default=5555, help='Frontend port number (clients)')
    parser.add_argument('--backend', type=int, default=5556, help='Backend port number (workers)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    main(args)