import argparse
import os
import multiprocessing
import jumpstreet
import time


def start_process_from_config(target, config):
    process = multiprocessing.Process(target=target, args=config)
    process.daemon = True
    process.start()
    return process


class DemoController():
    def __init__(self, broker, detection, tracking, frontend) -> None:
        self.processes = []
        self.broker = broker
        self.detection = detection
        self.tracking = tracking
        self.frontend = frontend

    def start(self):
        """Start up all the processes
        
        The order of starting is important in some cases.
        e.g., if using interprocess communication, the binding
        socket MUST be started first.
        """
        self.processes.append(jumpstreet.broker.start_process_from_config(jumpstreet.broker.main, self.broker))
        self.processes.append(jumpstreet.detection.start_process_from_config(jumpstreet.detection.main, self.detection))
        self.processes.append(jumpstreet.tracking.start_process_from_config(jumpstreet.tracking.main, self.tracking))
        self.processes.append(jumpstreet.frontend.start_process_from_config(jumpstreet.frontend.simple.main, self.frontend))
        while True:
            time.sleep(0.1)

    def end(self):
        for process in self.processes:
            process.join(timeout=0)
            process.kill()


def config_path(config):
    return os.path.join(os.path.dirname(__file__), '../configs', config)


def main(args):
    """Start up the processes with configs"""
    broker = jumpstreet.utils.config_as_namespace(args.broker_config)
    detection = jumpstreet.utils.config_as_namespace(args.detection_config)
    tracking = jumpstreet.utils.config_as_namespace(args.tracking_config)
    frontend = jumpstreet.utils.config_as_namespace(args.frontend_config)
    control = DemoController(broker=broker,
        detection=detection, tracking=tracking, frontend=frontend)
    try:
        control.start()
    except:
        control.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the demo platform controllers."
    )
    parser.add_argument(
        "--broker_config", default='broker/default.yml'
    )
    parser.add_argument(
        "--detection_config", default='detection/default.yml'
    )
    parser.add_argument(
        "--tracking_config", default='tracking/default.yml'
    )
    parser.add_argument(
        "--frontend_config", default='frontend/default.yml'
    )
    args = parser.parse_args()
    main(args)