#!/usr/bin/env python3

import argparse
import logging
import zmq
import multiprocessing
import random
from time import sleep
from jumpstreet import client as jclient


class SensorDataReplayer():
    """Replays sensor data from a folder"""

    def __init__(self, context, HOST, PORT, identifier, send_dir) -> None:
        self.identifier = identifier
        self.interface = jclient.ClientWithRouter(context, HOST=HOST, PORT=PORT, identifier=identifier)

    def send(self):
        # -- load data
        data = "test-{}".format(random.randint(0, 200)).encode("ascii")

        # -- send data
        self.interface.print('sending data...', end='')
        self.interface.socket.send(data)
        print('done')
        
        # -- acknowledge
        self.interface.print('waiting for acknowledge...', end='')
        self.interface.receive_acknowledge()
        print('done')

    def close(self):
        self.interface.close()


def start_client(task, *args):
    """Starting a client using multiproc"""
    process = multiprocessing.Process(target=task, args=args)
    process.daemon = True
    process.start()


def main_single(HOST, PORT, identifier, send_rate, send_dir):
    """Runs sending on a single client"""
    context = zmq.Context.instance()
    replayer = SensorDataReplayer(context, HOST=HOST, PORT=PORT,
        identifier=identifier, send_dir=send_dir)
    send_dt = 1./send_rate
    try:
        while True:
            replayer.send()
            sleep(send_dt)
    except Exception as e:
        logging.warning(e)
    finally:
        replayer.close()


def main(args):
    """Run sensor replayer clients"""
    for i in range(args.nclients):
        start_client(main_single, args.host, args.port, i, args.send_rate, args.send_dir)
    while True:
        sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Initialize sensor replayer client')
    parser.add_argument('-n' , '--nclients', type=int, default=1, help='Number of clients')
    parser.add_argument('--host', default='localhost', type=str, help='Hostname to connect to')
    parser.add_argument('--port', default=5555, type=int, help='Port to connect to server/broker')
    parser.add_argument('--send_rate', default=10, type=int, help='Replay rate for sensor data')
    parser.add_argument('--send_dir', type=str, default='./data/TUD-Campus/img1', help='Directory for data replay')

    args = parser.parse_args()
    main(args)