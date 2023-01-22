import zmq
import numpy as np


class BaseClass():
    def __init__(self, name, identifier) -> None:
        self.name = name
        self.identifier = identifier
        self.frontend = None
        self.backend = None

    @property
    def frontend(self):
        return self._frontend

    @frontend.setter
    def frontend(self, frontend):
        self._frontend = frontend

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        self._backend = backend

    def close(self):
        if self.frontend is not None:
            self.frontend.close()
        if self.backend is not None:
            self.backend.close()

    def print(self, msg, end='', flush=True):
        try:
            name = self.NAME
        except AttributeError as e:
            name = self.name
        print(f'::{name}-{self.identifier}::{msg}', end=end, flush=True)


def init_some_end(cls, context, end_type, pattern, HOST, PORT, BIND=False, subopts=None):
    socket = context.socket(pattern)
    if BIND:
        pstring = f'creating {end_type} bind with {PORT}...'
        if cls is not None:
            cls.print(pstring, end='')
        else:
            print(pstring, end='')
        socket.bind(f"tcp://*:{PORT}")
        print('done')
    else:
        pstring = f'creating {end_type} connection with {HOST}:{PORT}...'
        if cls is not None:
            cls.print(pstring, end='')
        else:
            print(pstring)
        socket.connect(f"tcp://{HOST}:{PORT}")
        print('done')
    if pattern == zmq.SUB:
        assert subopts is not None
        socket.setsockopt(zmq.SUBSCRIBE, subopts)
    return socket


def send_array_reqrep(zmq_socket, msg, array):
    """Sends OpenCV array and msg to hub computer in REQ/REP mode
    Arguments:
        msg: text message or array name.
        array: OpenCV array to send to hub.
    Returns:
        A text reply from hub.
    """

    if array.flags['C_CONTIGUOUS']:
        # if array is already contiguous in memory just send it
        zmq_socket.send_array(array, msg, copy=False)
    else:
        # else make it contiguous before sending
        array = np.ascontiguousarray(array)
        zmq_socket.send_array(array, msg, copy=False)
    hub_reply = zmq_socket.recv()  # receive the reply message
    return hub_reply


def send_array_pubsub(zmq_socket, msg, array):
    """Sends OpenCV array and msg hub computer in PUB/SUB mode. If
    there is no hub computer subscribed to this socket, then array and msg
    are discarded.
    Arguments:
        msg: text message or array name.
        array: OpenCV array to send to hub.
    Returns:
        Nothing; there is no reply from hub computer in PUB/SUB mode
    """

    if array.flags['C_CONTIGUOUS']:
        # if array is already contiguous in memory just send it
        zmq_socket.send_array(array, msg, copy=False)
    else:
        # else make it contiguous before sending
        array = np.ascontiguousarray(array)
        zmq_socket.send_array(array, msg, copy=False)


def send_jpg_reqrep(zmq_socket, msg, jpg_buffer):
    """Sends msg text and jpg buffer to hub computer in REQ/REP mode.
    Arguments:
        msg: array name or message text.
        jpg_buffer: bytestring containing the jpg array to send to hub.
    Returns:
        A text reply from hub.
    """

    zmq_socket.send_jpg(msg, jpg_buffer, copy=False)
    hub_reply = zmq_socket.recv()  # receive the reply message
    return hub_reply


def send_jpg_pubsub(zmq_socket, msg, jpg_buffer):
    """Sends msg text and jpg buffer to hub computer in PUB/SUB mode. If
    there is no hub computer subscribed to this socket, then array and msg
    are discarded.
    Arguments:
        msg: array name or message text.
        jpg_buffer: bytestring containing the jpg array to send to hub.
    Returns:
        Nothing; there is no reply from the hub computer in PUB/SUB mode.
    """

    zmq_socket.send_jpg(msg, jpg_buffer, copy=False)


def recv_array(zmq_socket, copy=False):
    """Receives OpenCV array and text msg.
    Arguments:
        copy: (optional) zmq copy flag.
    Returns:
        msg: text msg, often the array name.
        array: OpenCV array.
    """

    msg, array = zmq_socket.recv_array(copy=False)
    return msg, array


def recv_jpg(zmq_socket, copy=False):
    """Receives text msg, jpg buffer.
    Arguments:
        copy: (optional) zmq copy flag
    Returns:
        msg: text message, often array name
        jpg_buffer: bytestring jpg compressed array
    """

    msg, jpg_buffer = zmq_socket.recv_jpg(copy=False)
    return msg, jpg_buffer
