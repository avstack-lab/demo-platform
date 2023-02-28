import PySpin

# Connect to the first camera on the system
system = PySpin.System.GetInstance()
cam_list = system.GetCameras()
cam = cam_list.GetByIndex(0)
cam.Init()

# Set the camera properties here
cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
cam.AcquisitionFrameRate.SetValue(30)
cam.Width.SetValue(640)
cam.Height.SetValue(480)

# Start the camera streaming to a localhost port using zmq
context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.bind('tcp://127.0.0.1:5555')

# Start the acquisition and streaming loop
cam.BeginAcquisition()
while True:
    image = cam.GetNextImage()
    image_data = image.GetData()
    socket.send(image_data)
    image.Release()

# Stop the acquisition and close the camera
cam.EndAcquisition()
cam.DeInit()
del cam
cam_list.Clear()
system.ReleaseInstance()