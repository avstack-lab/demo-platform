import json

import numpy as np
import zmq


class SerializingSocket(zmq.Socket):
    """Numpy array serialization methods.
    Modelled on PyZMQ serialization examples.
    Used for sending / receiving OpenCV images, which are Numpy arrays.
    Also used for sending / receiving jpg compressed OpenCV images.
    Can also be used to send any built-in python data types
    """

    def send_array(self, A, msg="NoName", flags=0, copy=True, track=False):
        """Sends a numpy array with metadata and text message.

        Sends a numpy array with the metadata necessary for reconstructing
        the array (dtype,shape). Also sends a text msg, often the array or
        image name.
        Arguments:
            A: numpy array or OpenCV image.
            msg: (optional) array name, image name or text message.
            flags: (optional) zmq flags.
            copy: (optional) zmq copy flag.
            track: (optional) zmq track flag.
        """

        md = dict(
            msg=msg,
            dtype=str(A.dtype),
            shape=A.shape,
        )
        self.send_json(md, flags | zmq.SNDMORE)
        return self.send(A, flags, copy=copy, track=track)

    def send_jpg(self, msg="NoName", jpg_buffer=b"00", flags=0, copy=True, track=False):
        """Send a jpg buffer with a text message.
        Sends a jpg bytestring of an OpenCV image.
        Also sends text msg, often the image name.
        Arguments:
            msg: image name or text message.
            jpg_buffer: jpg buffer of compressed image to be sent.
            flags: (optional) zmq flags.
            copy: (optional) zmq copy flag.
            track: (optional) zmq track flag.
        """

        md = dict(
            msg=msg,
        )
        self.send_json(md, flags | zmq.SNDMORE)
        return self.send(jpg_buffer, flags, copy=copy, track=track)

    def send_array_envelope(
        self, env_to, env_from, msg, array, flags=0, copy=True, track=False
    ):
        """Send an array, with a routing envelope

        Always remember your empty frame!

        Arguments:
            Same as before...
        """
        md = dict(
            msg=msg,
            dtype=str(array.dtype),
            shape=array.shape,
        )
        md = json.dumps(md, indent=2).encode("utf-8")
        if array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        self.send_multipart(
            [env_to, b"", env_from, md, array], flags=flags, copy=copy, track=track
        )

    def recv_array_envelope(self, flags=0, copy=True, track=False):
        """Receives an array wrapped in an evelope i.e. for a router"""
        client, _, metadata, array = self.recv_multipart(
            flags=flags, copy=copy, track=track
        )
        metadata = json.loads(metadata.decode("utf-8"))
        array = np.frombuffer(array, dtype=metadata["dtype"])
        return (client, metadata, array)

    def recv_array_multipart(self, flags=0, copy=True, track=False):
        """Same as envelope but without the empty frame i.e. not on router"""
        client, metadata, array = self.recv_multipart(
            flags=flags, copy=copy, track=track
        )
        metadata = json.loads(metadata)  # .decode("utf-8")
        array = np.frombuffer(array, dtype=metadata["dtype"])
        return (client, metadata, array)

    def recv_array(self, flags=0, copy=True, track=False):
        """Receives a numpy array with metadata and text message.
        Receives a numpy array with the metadata necessary
        for reconstructing the array (dtype,shape).
        Returns the array and a text msg, often the array or image name.
        Arguments:
            flags: (optional) zmq flags.
            copy: (optional) zmq copy flag.
            track: (optional) zmq track flag.
        Returns:
            msg: image name or text message.
            A: numpy array or OpenCV image reconstructed with dtype and shape.
        """

        md = self.recv_json(flags=flags)
        msg = self.recv(flags=flags, copy=copy, track=track)
        A = np.frombuffer(msg, dtype=md["dtype"])
        return (md["msg"], A.reshape(md["shape"]))

    def recv_jpg(self, flags=0, copy=True, track=False):
        """Receives a jpg buffer and a text msg.
        Receives a jpg bytestring of an OpenCV image.
        Also receives a text msg, often the image name.
        Arguments:
            flags: (optional) zmq flags.
            copy: (optional) zmq copy flag.
            track: (optional) zmq track flag.
        Returns:
            msg: image name or text message.
            jpg_buffer: bytestring, containing jpg image.
        """

        md = self.recv_json(flags=flags)  # metadata text
        jpg_buffer = self.recv(flags=flags, copy=copy, track=track)
        return (md["msg"], jpg_buffer)


class SerializingContext(zmq.Context):
    _socket_class = SerializingSocket
