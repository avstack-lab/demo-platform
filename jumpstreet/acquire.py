"""
Author: Nate Zelter
Date: February 2023
"""


import PySpin # must run with python3
import zmq
import sys
import numpy as np
import cv2

STOP_KEY = 'q'

def main(camera_configs):

    ## Extract camera configuration settings ##
    cam_name = camera_configs.get('name', 'NA')
    cam_serial = camera_configs.get('serial', 'NA')
    cam_ip = camera_configs.get('ip', 'NA')  
    cam_width_px = int(camera_configs.get('width_px', 0))  
    cam_height_px = int(camera_configs.get('height_px', 0))  
    cam_fps = int(camera_configs.get('fps', 0))
    cam_frame_size_bytes = int(camera_configs.get('frame_size_bytes', 0))


    ## Connect to camera
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam_i = cam_list.GetBySerial(cam_serial)
    try:
        cam_i.Init()
        print(f'Successfully connected to {cam_name} via serial number')
    except:
        print(f"Unable to connect to {cam_name} via serial number")
        return 1


    ## Set the camera properties here
    cam_i.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    cam_i.Width.SetValue(cam_width_px)
    cam_i.Height.SetValue(cam_height_px)
    cam_i.AcquisitionFrameRateEnable.SetValue(True) #enable changes to FPS
    cam_i.AcquisitionFrameRate.SetValue(cam_fps) # max is 24fps for FLIR BFS


    # Start the camera streaming to a localhost port using zmq
    # context = zmq.Context()
    # socket = context.socket(zmq.PUSH)
    # socket.bind('tcp://127.0.0.1:7555')


    ## Start the acquisition and streaming loop
    cam_i.BeginAcquisition()
    print("Startng camera acquisition")
    while True:
        image_ptr = cam_i.GetNextImage()
        image_dimensions = (cam_height_px, cam_width_px)
        image_array = np.frombuffer(image_ptr.GetData(), dtype=np.uint8).reshape(image_dimensions)
        # print(f"image array: {sys.getsizeof(image_array)}\n arrray len: {len(image_array)}")


        # frame = cv2.cvtColor(image_array, cv2.COLOR_BayerBG2BGR)  # for RGB camera demosaicing
        # print(f"color image array: {sys.getsizeof(frame)}\n color image array len: {len(frame)}")

        # frame_show = cv2.resize(frame, None, fx=0.25, fy=0.25)
        image_show = cv2.resize(image_array, None, fx=1, fy=1)
        cv2.imshow(f"Press {STOP_KEY} to quit", image_show)
        key = cv2.waitKey(30)
        if key == ord(STOP_KEY):
            print('Received STOP_KEY signal')
            break


        # socket.send(image_array)
        image_ptr.Release()

        

    # Stop the acquisition and close the camera and close socket
    cam_i.EndAcquisition()
    cam_i.DeInit()
    del cam_i
    cam_list.Clear()
    system.ReleaseInstance()
    # socket.close()
    return 0


if __name__ == '__main__':
    # camera_serials = ['22395929', '22395953']
    # camera_ips = ['192.168.1.1', '192.168.1.2']

    config = {
        'camera_1': {
            'name': 'camera_1',
            'type': 'FLIR-BFS-50S50C',
            'serial': '22395929',
            'ip': '192.168.1.1',
            'width_px': '480',
            'height_px': '640',
            'fps': '10',
            'frame_size_bytes': '307200'  
            },
        'camera_2': {
            'name': 'camera_2',
            'type': 'FLIR-BFS-50S50C',
            'serial': '22395953',
            'ip': '192.168.1.2',
            'width_px': '480',
            'height_px': '640',
            'fps': '10',
            'frame_size_bytes': '307200'  
            }
        }
    

    status = main(config['camera_1'])
    print(f"Completed script with {status} error(s)...")
    