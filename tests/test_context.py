import zmq

from jumpstreet import context as jcontext


def test_serializing_init():
    context = jcontext.SerializingContext()
    socket = context.socket(zmq.REQ)
    assert socket is not None
