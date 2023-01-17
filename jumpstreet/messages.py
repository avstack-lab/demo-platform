import jsonpickle


def encode(cls, binary=True):
    msg = jsonpickle.encode(cls)
    return bytearray(msg, encoding='utf8') if binary else msg


def decode(msg, binary=True):
    msg = bytes.decode(msg) if binary else msg
    return jsonpickle.decode(msg)


class Message():
    def encode(self, binary=True):
        return encode(self, binary=binary)


class SensorStartup(Message):
    def __init__(self, sensor_name) -> None:
        self.sensor_name = sensor_name


class SensorStartupReply(Message):
    def __init__(self, sensor_name, IM_IN_PORT, ANALY_OUT_PORT) -> None:
        self.IM_IN_PORT = IM_IN_PORT
        self.ANALY_OUT_PORT = ANALY_OUT_PORT
        self.sensor_name = sensor_name


class SensorAnalysisOutput(Message):
    def __init__(self, sensor_name, detections, tracks, trigger) -> None:
        self.sensor_name = sensor_name
        self.detections = detections
        self.tracks = tracks
        self.trigger = trigger


class UnknownMessageError(Exception):
    def __init__(self, unknown_mess, *args):
        message = "Unknown message type!"
        self.unknown_mess = unknown_mess         
        super(UnknownMessageError, self).__init__(message, unknown_mess, *args) 