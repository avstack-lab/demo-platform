import zmq
from jumpstreet import utils as jutils


def test_init_some_end():
    context = zmq.Context.instance()
    socket = jutils.init_some_end(cls=None, end_type='frontend',
        context=context, pattern=zmq.REQ, HOST='*', PORT=5555, BIND=True)
    assert socket is not None