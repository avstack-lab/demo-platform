import zmq

from jumpstreet import utils as jutils


def test_init_some_end():
    context = zmq.Context.instance()
    socket = jutils.init_some_end(
        cls=None,
        context=context,
        end_type="frontend",
        pattern=zmq.REQ,
        TRANSPORT="ipc",
        HOST="*",
        PORT=5555,
        BIND=True,
    )
    assert socket is not None