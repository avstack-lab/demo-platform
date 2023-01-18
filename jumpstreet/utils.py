import zmq


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
        cls.print(f'creating {end_type} bind with {PORT}', end='\n')
        socket.bind(f"tcp://*:{PORT}")
    else:
        cls.print(f'creating {end_type} connection with {HOST}:{PORT}', end='\n')
        socket.connect(f"tcp://{HOST}:{PORT}")
    if pattern == zmq.SUB:
        assert subopts is not None
        socket.setsockopt(zmq.SUBSCRIBE, subopts)
    return socket