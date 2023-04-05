"""
Author: Nate Zelter
Date: April 2023

"""


import argparse
import os
import subprocess
import time


AVAILABLE_COMMANDS = ["simple-flir-old", "camera-1", "camera-1-verbose"]


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute multiple Python files concurrently."
    )
    parser.add_argument(
        "command", choices=AVAILABLE_COMMANDS, help="the command to run"
    )
    args = parser.parse_args()
    main(args)
