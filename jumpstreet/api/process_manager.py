#!/usr/bin/env python3

import argparse
import logging
from time import sleep
import zmq
import jumpstreet.messages as jmess


def printhere(msg, end='\n'):
    print('::pm::' + msg, end=end)


class ProcessManager:
    def __init__(self, PORT=5555, max_sensor_proc=3) -> None:
        self.max_sensor_proc = max_sensor_proc
        printhere(f"establishing tcp on port {PORT}...", end='')
        self.PORT = PORT
        self.pm_zmq_context = zmq.Context()
        self.pm_zmq_socket = self.pm_zmq_context.socket(zmq.REP)
        self.pm_zmq_socket.bind(f"tcp://*:{args.port}")
        print("done")
        self.sensor_zmq_ports = {}
        self.last_PORT = PORT

    def poll(self):
        """Wait in a blocking-fashion for the next message"""
        json_msg = self.pm_zmq_socket.recv()
        msg = jmess.decode(json_msg)
        if isinstance(msg, jmess.SensorStartup):
            if msg.sensor_name in self.sensor_zmq_ports.keys():
                raise RuntimeError(f'Sensor name {msg.sensor_name} is already established')
            else:
                next_PORT = self.last_PORT + 1
                while next_PORT in self.sensor_zmq_ports.keys():
                    next_PORT += 1
                printhere(f'telling sensor to initialize on data port {next_PORT}...', end='')
                self.sensor_zmq_ports[msg.sensor_name] = next_PORT
                self.pm_zmq_socket.send(jmess.encode(
                    jmess.SensorStartupReply(msg.sensor_name, next_PORT)))
                self.last_PORT = next_PORT
                print('done')
                printhere
        else:
            logging.warning(jmess.UnknownMessageError(msg))


def main(args):
    procm = ProcessManager(PORT=args.port, max_sensor_proc=args.max_sensor_proc)
    printhere("Waiting for connections...")
    while True:
        procm.poll()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start up the process manager")
    parser.add_argument(
        "--max_sensor_proc",
        default=3,
        type=int,
        help="Maximum number of sensor processes to be running",
    )
    parser.add_argument(
        "--port",
        default=5555,
        type=int,
        help="Port number for TCP connections to PM"
    )

    args = parser.parse_args()
    main(args)
