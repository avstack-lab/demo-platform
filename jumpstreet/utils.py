import json
import os
import sys
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import yaml
import zmq


class BaseClass:
    def __init__(self, name, identifier, verbose=False, debug=False) -> None:
        self.name = name
        self.identifier = identifier
        self.verbose = verbose
        self.debug = debug
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

    def print(self, msg, end="", flush=True):
        try:
            name = self.NAME
        except AttributeError as e:
            name = self.name
        print(f"::{name}-{self.identifier}::{msg}", end=end, flush=flush)
        sys.stdout.flush()


class TimeMonitor(BaseClass):
    def __init__(self, maxlen=10) -> None:
        super().__init__("time-monitor", 0, verbose=True, debug=False)
        self.dt_history = deque([], maxlen=maxlen)
        self.last_t = None

    def trigger(self):
        if self.last_t is None:
            self.last_t = time.time()
        else:
            new_t = time.time()
            self.dt_history.append(new_t - self.last_t)
            if len(self.dt_history) > 3:
                self.last_t = new_t
                dt = np.mean(self.dt_history)
                fps = 1.0 / dt
                std = np.std([1.0 / dt for dt in self.dt_history])
                self.print(f"FPS: {fps:4.2f},   FPS std: {1./std:2.3f}", end="\r")


class SocketConfig:
    def __init__(self, config) -> None:
        self.host = config.host
        self.port = config.port
        self.bind = config.bind


def config_as_namespace(config_file):
    config_path = os.path.join(os.path.dirname(__file__), "configs", config_file)
    if not os.path.exists(config_path):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", config_file
        )
        if not os.path.exists(config_path):
            raise FileNotFoundError(config_file)
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    def load_object(dct):
        return SimpleNamespace(**dct)

    ns = json.loads(json.dumps(config), object_hook=load_object)
    return ns


def init_some_end(
    cls, context, end_type, pattern, TRANSPORT, HOST, PORT, BIND=False, subopts=None
):
    socket = context.socket(pattern)
    if BIND:
        pstring = f"creating {end_type} bind with {PORT}..."
        if cls is not None:
            cls.print(pstring, end="")
        else:
            print(pstring, end="")
        if TRANSPORT == "tcp":
            socket.bind(f"{TRANSPORT}://*:{PORT}")
        elif TRANSPORT == "ipc":
            socket.bind(f"{TRANSPORT}://{HOST}:{PORT}")
        else:
            raise NotImplementedError
        print("done")
    else:
        pstring = f"creating {end_type} connection with {HOST}:{PORT}..."
        if cls is not None:
            cls.print(pstring, end="")
        else:
            print(pstring, end="")
        socket.connect(f"{TRANSPORT}://{HOST}:{PORT}")
        print("done")
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

    if array.flags["C_CONTIGUOUS"]:
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

    if array.flags["C_CONTIGUOUS"]:
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
