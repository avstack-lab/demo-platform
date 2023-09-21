import argparse
import multiprocessing
import os
import time
import shutil

import jumpstreet
from jumpstreet.utils import config_as_namespace


def start_process_from_config(target, config):
    process = multiprocessing.Process(target=target, args=[config])
    process.daemon = False
    process.start()
    return process


class Controller:
    def __init__(self, broker, detection, tracking, display) -> None:
        self.processes = []
        self.broker = broker
        self.detection = detection
        self.tracking = tracking
        self.display = display

    def start(self):
        """Start up all the processes

        The order of starting is important in some cases.
        e.g., if using interprocess communication, the binding
        socket MUST be started first.
        """
        self.processes.append(
            start_process_from_config(jumpstreet.broker.main, self.broker)
        )
        self.processes.append(
            start_process_from_config(jumpstreet.tracking.main, self.tracking)
        )
        self.processes.append(
            start_process_from_config(jumpstreet.detection.main, self.detection)
        )
        self.processes.append(
            start_process_from_config(jumpstreet.display.simple.main, self.display)
        )
        while True:
            time.sleep(0.1)

    def end(self):
        for process in self.processes:
            process.join(timeout=0)
            process.kill()


def config_path(config):
    return os.path.join(os.path.dirname(__file__), "../configs", config)


def main(config):
    """Start up the processes with configs"""
    if os.path.exists('profiles'):
        shutil.rmtree('profiles')
    control = Controller(
        broker=config_as_namespace(config.modules.broker.config),
        detection=config_as_namespace(config.modules.detection.config),
        tracking=config_as_namespace(config.modules.tracking.config),
        display=config_as_namespace(config.modules.display.config),
    )
    try:
        control.start()
    except Exception as e:
        control.end()
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the demo platform controller.")
    parser.add_argument("--config", help="Controller configuration parameters")
    args = parser.parse_args()
    config = config_as_namespace(args.config)
    main(config)
