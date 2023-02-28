import zmq
import numpy as np



def main(args):
    # initialize ZeroMQ context and socket
    socket_address = args['socket_address']


    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:5555")
    socket.setsockopt(zmq.SUBSCRIBE, b'')


    while True:
        data = socket.recv()
        print(np.frombuffer(data, dtype=np.uint8))


if __name__ == '__main__':
    args = {
        'context':'<zmq.Context()>',
        'socket_address': ('127.0.0.1', 5555),
        'info': 'blah'
    }
    main(args)
