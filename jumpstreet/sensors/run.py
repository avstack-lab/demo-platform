import argparse

from jumpstreet.context import SerializingContext
from jumpstreet.sensors.camera import PySpinCamera, ReplayCamera
from jumpstreet.sensors.radar import TiRadar
from jumpstreet.utils import config_as_namespace


def main(config, sensor_id):
    # -- find sensor class
    context = SerializingContext()
    if config.sensor_class == "ReplayCamera":
        SensorClass = ReplayCamera
    elif config.sensor_class == "PySpinCamera":
        SensorClass = PySpinCamera
    elif config.sensor_class == "TiRadar":
        SensorClass = TiRadar
    else:
        raise NotImplementedError(args.sensor_type)

    # -- init sensor class
    sensor = SensorClass(
        context,
        config.backend,
        config,
        sensor_id,
        verbose=config.verbose,
        debug=config.debug,
    )

    # -- initialize and run sensor
    sensor.initialize()
    if config.verbose:
        print("Sensor successfully initialized in sensor.py")
    sensor.start_capture()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Initialize a Sensor")
    parser.add_argument("--config", help="Select the configuration to apply")
    parser.add_argument(
        "--sensor_id", default="camera_1", help="Identifier of the camera"
    )
    args = parser.parse_args()
    config = config_as_namespace(args.config)
    main(config, args.sensor_id)
