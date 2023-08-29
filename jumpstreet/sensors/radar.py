import time

import numpy as np
import rad
from avstack.calibration import Calibration
from avstack.geometry.transformations import matrix_cartesian_to_spherical

from .base import Sensor


class Radar(Sensor):
    NAME = "radar-sensor"

    def initialize(self):
        self.radar = rad.Radar(
            config_file_name=self.config.cfg_file,
            translate_from_JSON=False,
            enable_serial=True,
            CLI_port=self.config.CLI_port,
            Data_port=self.config.DATA_port,
            enable_plotting=False,
            jupyter=False,
            data_file=None,
            refresh_rate=self.config.refresh_rate,
            verbose=False,
        )

    def _send_radar_data(self, razelrrt, ts, frame):
        # -- send across comms channel
        msg = {
            "timestamp": ts,
            "frame": frame,
            "identifier": self.identifier,
            "extrinsics": self.calibration.format_as_string(),
        }
        self.backend.send_array(razelrrt, msg, False)
        if self.debug:
            self.print(
                f"sending data, frame: {msg['frame']:4d}, timestamp: {msg['timestamp']:.4f}",
                end="\n",
            )


class TiRadar(Radar):
    def __init__(
        self,
        context,
        backend,
        config,
        identifier,
        min_range=1.0,
        verbose=False,
        debug=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            context,
            backend,
            config,
            identifier,
            verbose,
            debug,
        )
        self.radar = None
        self.frame = 0
        self.min_range = min_range
        self.calibration = Calibration(self.extrinsics)

    def start_capture(self):
        self.radar.start()
        while True:
            try:
                # -- read from serial port
                time.sleep(self.radar.refresh_delay)
                xyzrrt = self.radar.read_serial()
                if xyzrrt is None:
                    continue
                razelrrt = matrix_cartesian_to_spherical(
                    np.array(
                        [xyzrrt[:, 1], -xyzrrt[:, 0], xyzrrt[:, 2], xyzrrt[:, 3]]
                    ).T
                )
                razelrrt = razelrrt[razelrrt[:, 0] > self.min_range, :]
                timestamp = time.time()
                self._send_radar_data(razelrrt, timestamp, self.frame)
                self.frame += 1
                self.time_monitor.trigger()
            except KeyboardInterrupt:
                self.radar.streamer.stop_serial_stream()
                if self.verbose or self.debug:
                    self.print("Radar.stream_serial: stopping serial stream")
                break
