import time

import rad
from avstack.geometry.transformations import matrix_cartesian_to_spherical

from .base import Sensor


class Radar(Sensor):
    NAME = "radar-sensor"

    def initialize(self):
        self.radar = rad.Radar(
            config_file_name=self.configuration["config_file_name"],
            translate_from_JSON=False,
            enable_serial=True,
            CLI_port=self.configuration["CLI_port"],
            Data_port=self.configuration["Data_port"],
            enable_plotting=False,
            jupyter=False,
            data_file=None,
            refresh_rate=self.configuration["refresh_rate"],
            verbose=False,
        )


class TiRadar(Radar):
    def __init__(
        self,
        context,
        backend,
        config,
        identifier,
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

    def start_capture(self):
        self.radar.start()
        t0 = time.time()
        while True:
            try:
                # -- read from serial port
                time.sleep(self.radar.refresh_delay)
                xyzrrt = self.radar.read_serial()
                if xyzrrt is None:
                    continue
                razelrrt = xyzrrt.copy()
                razelrrt[:, :3] = matrix_cartesian_to_spherical(xyzrrt[:, :3])

                # -- send across comms channel
                timestamp = round(time.time() - t0, 9)
                msg = {
                    "timestamp": timestamp,
                    "frame": self.frame,
                    "identifier": self.identifier,
                    "extrinsics": [0, 0, 0, 0, 0],
                }
                self.backend.send_array(razelrrt, msg, False)
                if self.debug:
                    self.print(
                        f"sending data, frame: {msg['frame']:4d}, timestamp: {msg['timestamp']:.4f}",
                        end="\n",
                    )
                self.frame += 1
            except KeyboardInterrupt:
                self.radar.streamer.stop_serial_stream()
                if self.verbose or self.debug:
                    print("Radar.stream_serial: stopping serial stream")
                break
