import numpy as np
import torch
import torch_model
import cv2
import camera_tools as ct
from FableAPI.fable_init import api
from utils import initialize_camera, initialize_robot
from time import sleep

cam = ct.prepare_camera()
print(cam.isOpened())  # False
i = 0

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
model = torch_model.Net(4, 50, 2)
model.load_state_dict(torch.load('closed_loop_trained_model.pth'))


# dummy class for targets
class CoordinateStore:
    def __init__(self):
        self.point = None
        self.new = False

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
            self.point = [x, y]
            self.new = True


t = [40, 40]

# Instantiate class
coordinateStore1 = CoordinateStore()

# as alternative you can set prior targets

cv2.namedWindow("test")
cv2.setMouseCallback('test', coordinateStore1.select_point)

inp = None
model.eval()

while True:
    frame = ct.capture_image(cam)
    xy = ct.locate(frame)

    cv2.imshow("test", frame)
    k = cv2.waitKey(100)
    if k == 27:
        break

    #print(coordinateStore1.point)
    # get the prediction
    if coordinateStore1.new:
        print('NEW POINT!!!')
        with torch.no_grad():
            inp = torch.tensor([coordinateStore1.point]).float()
            inp = (inp - 200) / 200

        coordinateStore1.new = False

    # frame = ct.capture_image(cam)
    # cv2.imshow("test", frame)
    # cv2.waitKey(1)
    xy = ct.locate(frame)
    if xy[0] is not None and inp is not None:
        print(xy)
        t = [api.getPos('X', module), api.getPos('Y', module)]
        current_xy = torch.tensor([xy]).float()
        current_xy = (current_xy - 200) / 200
        if np.linalg.norm(inp.numpy()[0] - current_xy.numpy()[0])*200 < 5:
            pass
        else:
            with torch.no_grad():
                m_input = torch.tensor([np.append(inp - current_xy, np.divide(t, 90))]).float()
                print(m_input)
                outp = model(m_input)
                t = outp.numpy()[0] * 90
                print(t)
            # t = [t[0] + dt[0], t[1] + dt[1]]
            t0 = [api.getPos('X', module), api.getPos('Y', module)]
            k = 0.8
            print("===")
            if np.linalg.norm(t - t0) > 4:
                target = k*(t - t0) + t0
                api.setPos(max(-90, min(90, target[0])), max(-90, min(90, target[1])), module)
            sleep(0.1)

    while api.getMoving('X', module) or api.getMoving('Y', module):
        sleep(0.1)
    # sleep(1.5)

print('Terminating')
api.terminate()
