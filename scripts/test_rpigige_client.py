from rpiasgige.client_api import Device

camera = Device("192.168.1.62", 4001)

camera.open()

ret, frame = camera.read()

cv.imshow("frame", frame)
cv.waitKey()

camera.release()
