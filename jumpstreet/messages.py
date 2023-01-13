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
    def __init__(self, sensor_name, DATA_PORT) -> None:
        self.DATA_PORT = DATA_PORT
        self.sensor_name = sensor_name


class UnknownMessageError(Exception):
    def __init__(self, unknown_mess, *args):
        message = "Unknown message type!"
        self.unknown_mess = unknown_mess         
        super(UnknownMessageError, self).__init__(message, unknown_mess, *args) 