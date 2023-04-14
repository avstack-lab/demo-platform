import argparse
import os
import subprocess
import time
import yaml

import jumpstreet
from jumpstreet.controllers import _Controller


class DemoController(_Controller):
    def __init__(self, broker, detection, tracking, frontend) -> None:
        super().__init__()
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
        self.processes.append(jumpstreet.broker.start_process_from_config(self.broker))
        self.processes.extend(jumpstreet.detection.start_process_from_config(self.detection))
        self.processes.extend(jumpstreet.tracking.start_process_from_config(self.tracking))
        self.processes.append(jumpstreet.frontend.start_process_from_config(self.frontend))



def config_path(config):
    return os.path.join(os.path.dirname(__file__), '../configs', config)


def main(args):
    """Start up the processes with configs"""
    broker = yaml.safe_load(config_path(args.broker_config))
    detection = yaml.safe_load(config_path(args.detection_config))
    tracking = yaml.safe_load(config_path(args.tracking_config))
    frontend = yaml.safe_load(config_path(args.frontend_config))
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












def main(args):
    if args.command == "simple-flir-old":
        cwd = os.getcwd()
        if cwd.split("/") == "jumpstreet":
            os.chdir("..")  # go from jumptstreet/ to demo-platform/
        commands = [
            "make data_broker",
            "make detection_workers",
            "make tracking_worker",
            "make frontend",
            "make flir",
        ]
    elif args.command == "camera-1":
        commands = [
            "poetry run python jumpstreet/broker.py \
                lb_with_xsub_extra_xpub \
                --frontend 5550 \
                --backend 5551 \
                --backend_other 5552",
            "poetry run python jumpstreet/object_detection.py \
                --n_image_workers 2 \
                --image_model fasterrcnn \
                --image_dataset coco-person \
                --n_radar_workers 0 \
                --radar_model none \
                --radar_dataset none \
                --image_threshold 0.5 \
                --radar_threshold 0.5 \
                --in_host localhost \
                --in_port 5551 \
                --out_host localhost \
                --out_port 5553",
            "poetry run python jumpstreet/object_tracking.py \
                --model sort \
                --in_host localhost \
                --in_port 5553 \
                --in_bind \
                --out_host localhost \
                --out_port 5554 \
                --out_bind",
            "poetry run python jumpstreet/frontend/simple.py \
                --host localhost \
                --port_tracks 5554 \
                --port_images=5552 \
                --display_cam_id camera_1 \
                --width 1224 \
                --height 1024",
            "poetry run python jumpstreet/sensor.py \
                --type camera-flir-bfs \
                --host 127.0.0.1 \
                --backend 5550 \
                --verbose \
                --resize_factor 4",
        ]

    if commands:
        processes = []
        try:
            for command in commands:
                processes.append(subprocess.Popen(command, shell=True))
                time.sleep(2)
            for process in processes:
                process.wait()
        except:
            for process in processes:
                pass
                # kill the process gracefully...


