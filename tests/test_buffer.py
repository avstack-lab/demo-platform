import jumpstreet
import numpy as np
from avstack.datastructs import DataContainer


def generate_data(dt_interval=0.05, n_data=100):
    data = {dt_interval*frame : DataContainer(frame, dt_interval*frame, np.random.randn(2,2)) for frame in range(n_data)}


def test_data_buffer():
    buffer = jumpstreet.buffer.BasicDataBuffer(identifier=0, max_size=30, verbose=False)
    for i in range(len(buffer[buffer.keys[0]])):
        print(i)


def test_delay_managed_buffer_event_driven():
    buffer = jumpstreet.buffer.DelayManagedDataBuffer(identifier=0, max_size=30, dt_delay=0.1, method='event-driven', verbose=False)


def test_delay_managed_buffer_real_time():
    buffer = jumpstreet.buffer.DelayManagedDataBuffer(identifier=0, max_size=30, dt_delay=0.1, method='real-time', verbose=False)
