import zmq
import random
import numpy as np
import sys
import select

STOP_KEY = 'q'
SEND_FREQ_HZ = 5

def main():
    # initialize ZeroMQ context and socket
    port = '5555'
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://127.0.0.1:{port}")

    print(f'Press "{STOP_KEY}" + <return> to stop')
    while True:
        # generate and send 1024B data frames
        data = np.random.bytes(1024)
        socket.send(data)

        # check for user input without blocking
        if select.select([sys.stdin], [], [], 0)[0]:
            """
            four arguments: 
                a list of file objects to monitor for input, 
                a list of file objects to monitor for output, 
                a list of file objects to monitor for exceptions, 
                and a timeout value in seconds.
            """
            if sys.stdin.read(1) == STOP_KEY:
                print('Received STOP key signal')
                print("Closing socket and exiting...")
                socket.close()
                break
    return 0
    

if __name__ == '__main__':
    main()
    context.term()