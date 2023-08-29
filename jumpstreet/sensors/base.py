import numpy as np
import zmq
from avstack.geometry import ReferenceFrame, GlobalOrigin3D

from jumpstreet.utils import BaseClass, TimeMonitor, init_some_end


class Sensor(BaseClass):
    """
    Sensor ZMQ Node, device agnostic
    Pattern: Data aquisition --> context.socket(zmq.PUB)
    """

    NAME = "generic-sensor"

    def __init__(
        self,
        context,
        backend,
        config,
        identifier,
        verbose=False,
        debug=False,
    ) -> None:
        super().__init__(self.NAME, identifier, verbose=verbose, debug=debug)
        self.config = config
        self.backend = init_some_end(
            self,
            context,
            "backend",
            zmq.PUB,
            backend.transport,
            backend.host,
            backend.port,
            BIND=backend.bind,
        )

        dx = np.array(
            [
                config.calibration.extrinsics.dx,
                config.calibration.extrinsics.dy,
                config.calibration.extrinsics.dz,
            ]
        )
        q = np.quaternion(
            *[
                config.calibration.extrinsics.qx,
                config.calibration.extrinsics.qy,
                config.calibration.extrinsics.qz,
            ]
        )
        self.reference = ReferenceFrame(dx, q, reference=GlobalOrigin3D)
        self.time_monitor = TimeMonitor()

    def initialize(self):
        raise NotImplementedError

    def start_capture(self):
        raise NotImplementedError
