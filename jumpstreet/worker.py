import zmq
import argparse
import uuid
from jumpstreet import pipeline as jpipe


class Worker():
    """Base class for a worker"""
    def __init__(self, context, HOST, PORT, identifier, pattern) -> None:
        self.NAME = f"worker-{identifier}"
        self.print(f'generating connection with {HOST}:{PORT}...', end='')
        self.socket = context.socket(pattern)
        self.socket.connect(f"tcp://{HOST}:{PORT}")
        print('done')

    def print(self, msg, end=''):
        print(f"::{self.NAME}::{msg}", end=end)

    def close(self):
        self.socket.close()
    

class WorkerWithRouter(Worker):
    """Worker that interacts with a router"""
    def __init__(self, context, HOST, PORT, identifier, pipeline) -> None:
        """
        Aguments:
        - context: the zmq context (one-per-process)
        - HOST: hostname to connect to
        - PORT: to connect to the router (backend)
        - identifier: unique identifier of this worker
        """
        super().__init__(context, HOST, PORT, identifier, pattern=zmq.REQ)

        # initialize worker pipeline
        self.pipeline = pipeline

        # tell broker we are ready to go
        self.socket.send(b"READY")

    def poll(self):
        address, empty, request = self.socket.recv_multipart()
        self.print('processing data with pipeline...', end='')
        pipe_out = self.pipeline(request)
        print('done')
        self.print(f'generated {pipe_out}', end='\n')
        self.socket.send_multipart([address, b"", b"OK"])


def init_worker(worker_type):
    if worker_type.lower() == 'workerwithrouter':
        return WorkerWithRouter
    else:
        raise NotImplemented(worker_type)


def main(args):
    """Run a worker"""
    context = zmq.Context.instance()
    identifier = int(uuid.uuid1()) % 2000
    pipeline = jpipe.init_pipeline(args.pipeline)
    worker = init_worker(args.worker)(context, HOST=args.host,
        PORT=args.port, identifier=identifier, pipeline=pipeline)
    try:
        while True:
            worker.poll()
    finally:
        worker.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Initialize a worker and pipeline')
    parser.add_argument('worker', choices=['workerwithrouter'], type=str, help="Type of worker")
    parser.add_argument('pipeline', default='append-world', type=str)
    parser.add_argument('--host', default='localhost', help='Hostname to connect to')
    parser.add_argument('--port', default=5556, help='Port to connect to server/broker')

    args = parser.parse_args()
    main(args)