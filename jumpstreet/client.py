#!/usr/bin/env python3

import zmq
import random
from time import sleep
import argparse
import uuid


class Client():
    """Base class for a client"""
    def __init__(self, context, HOST, PORT, identifier, pattern, verbose=False) -> None:
        self.NAME = f"client-{identifier}"
        self.verbose = verbose
        self.print(f'generating connection with {HOST}:{PORT}...', end='')
        self.socket = context.socket(pattern)
        self.socket.connect(f"tcp://{HOST}:{PORT}")
        print('done')

    def print(self, msg, end=''):
        print(f"::{self.NAME}::{msg}", end=end, flush=True)
    
    def close(self):
        self.socket.close()


class ClientWithRouter(Client):
    """Client that interacts with a router"""
    def __init__(self, context, HOST, PORT, identifier, verbose=False) -> None:
        """
        Aguments:
        - context: the zmq context (one-per-process)
        - HOST: hostname to connect to
        - PORT: to connect to the router (backend)
        - identifier: unique identifier of this worker
        """
        super().__init__(context, HOST, PORT, identifier, pattern=zmq.REQ, verbose=verbose)

        # tell broker we are ready to go (expect an immediate reply)
        print('sending startup message...', end='')
        self.socket.send(b"READY")
        print('done')
        self.print('waiting for acknowledge...', end='')
        self.receive_acknowledge()
        print('done')

    def receive_acknowledge(self):
        reply = self.socket.recv()
        assert reply.decode("ascii") == "OK"
        
    def send(self, data):
        self.print('sending data...', end='')
        self.socket.send(data)
        print('done')
        self.print('waiting for acknowledge...', end='')
        self.receive_acknowledge()
        print('done')


def init_client(client_type):
    if client_type.lower() == 'clientwithrouter':
        return ClientWithRouter
    else:
        raise NotImplementedError(client_type)


def main(args):
    """Run a dummy example"""
    context = zmq.Context.instance()
    message = b'hello'
    identifier = int(uuid.uuid1()) % 2000
    client = init_client(args.client)(context,
        HOST=args.host, PORT=args.port, identifier=identifier)
    try:
        while True:
            client.send(message)
            sleep(1 + random.random())
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Initialize a client')
    parser.add_argument('client', choices=['clientwithrouter'], type=str, help='Type of client to use')
    parser.add_argument('--host', default='localhost', type=str, help='Hostname to connect to')
    parser.add_argument('--port', type=int, default=5555, help='Port to connect client to server/broker')

    args = parser.parse_args()
    main(args)