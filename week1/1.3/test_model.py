import numpy as np
import torch
import torch_model
import cv2
import camera_tools as ct
from FableAPI.fable_init import api

cam = ct.prepare_camera()
print(cam.isOpened())  # False
i = 0

# Initialize the camera first.. waits for it to detect the green block
def initialize_camera(cam):
    while True:
        frame = ct.capture_image(cam)
        x, _ = ct.locate(frame)

        if x is not None:
            break

# Initialize the robot module
def initialize_robot(module=None):
    api.setup(blocking=True)
    # Find all the robots and return their IDs.
    print('Search for modules')
    moduleids = api.discoverModules()

    if module is None:
        module = moduleids[0]
    print('Found modules: ',moduleids)
    api.setPos(0,0, module)
    api.sleep(0.5)

    return module

initialize_camera(cam)
module = initialize_robot()

# Set move speed
speedX = 25
speedY = 25
api.setSpeed(speedX, speedY, module)

# Set accuracy
accurateX = 'HIGH'
accurateY = 'HIGH'
api.setAccurate(accurateX, accurateY, module)

# TODO Load the trained model
model = torch_model.MLPNet(2, 16, 2)
#model.load_state_dict(torch.load('model_file_path'))

# dummy class for targets
class CoordinateStore:
    def __init__(self):
        self.point = None
        self.new = False

    def select_point(self,event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                cv2.circle(frame,(x,y),3,(255,0,0),-1)
                self.point = [x,y]
                self.new = True


# Instantiate class
coordinateStore1 = CoordinateStore()

# as alternative you can set prior targets

cv2.namedWindow("test")
cv2.setMouseCallback('test', coordinateStore1.select_point)

while True:

    frame = ct.capture_image(cam)

    x, y = ct.locate(frame)

    cv2.imshow("test", frame)
    k = cv2.waitKey(500)
    if k == 27:
        break

    print(coordinateStore1.point)
    # get the prediction
    if coordinateStore1.new:
        with torch.no_grad():
            inp = torch.tensor([coordinateStore1.point]).float()
            outp = model(inp)
            t = outp.numpy()[0]
            print(t)
        api.setPos(t[0], t[1], module)
        coordinateStore1.new = False

print('Terminating')
api.terminate()

