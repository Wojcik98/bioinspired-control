import numpy as np
import torch
import torch_model
import cv2
import camera_tools as ct
from FableAPI.fable_init import api
from utils import initialize_camera, initialize_robot
from time import sleep
import pandas as pd

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

model = torch_model.Net(4, 50, 2)
model.load_state_dict(torch.load('closed_loop_trained_model.pth'))

t = [40, 40]

cv2.namedWindow("test")

model.eval()

np.random.seed(0)

runs = []
for i in range(100):
    print(f"{i}/100")
    run = []
    inp = [np.random.random()*(389-191) + 191, np.random.random()*(288-194)+194]
    inp = torch.tensor([inp]).float()
    inp = (inp - 200) / 200
    frame = ct.capture_image(cam)
    xy = ct.locate(frame)
    cv2.imshow("test", frame)
    k = cv2.waitKey(100)
    if k == 27:
        break
    while xy[0] is None:
        frame = ct.capture_image(cam)
        xy = ct.locate(frame)

        cv2.imshow("test", frame)
        k = cv2.waitKey(100)
        if k == 27:
            break

        xy = ct.locate(frame)
    current_xy = torch.tensor([xy]).float()
    current_xy = (current_xy - 200) / 200
    run.append(np.linalg.norm(inp - current_xy) * 200 + 200)
    for j in range(7):
        frame = ct.capture_image(cam)
        xy = ct.locate(frame)
        cv2.imshow("test", frame)
        k = cv2.waitKey(100)
        if k == 27:
            break
        while xy[0] is None:
            frame = ct.capture_image(cam)
            xy = ct.locate(frame)

            cv2.imshow("test", frame)
            k = cv2.waitKey(100)
            if k == 27:
                break

            xy = ct.locate(frame)

        t = [api.getPos('X', module), api.getPos('Y', module)]
        current_xy = torch.tensor([xy]).float()
        current_xy = (current_xy - 200) / 200

        with torch.no_grad():
            m_input = torch.tensor([np.append(inp - current_xy, np.divide(t, 90))]).float()
            outp = model(m_input)
            t = outp.numpy()[0] * 90
        # t = [t[0] + dt[0], t[1] + dt[1]]
        t0 = [api.getPos('X', module), api.getPos('Y', module)]
        k = 0.9

        target = k * (t - t0) + t0
        api.setPos(max(-90, min(90, target[0])), max(-90, min(90, target[1])), module)



        while api.getMoving('X', module) or api.getMoving('Y', module):
            sleep(0.1)

        run.append(np.linalg.norm(inp - current_xy) * 200 + 200)
        sleep(0.1)
        # sleep(1.5)
    runs.append(run)

print('Collected data:')
print(runs)

df = pd.DataFrame(runs)
df.to_csv('performance6-k09.csv', index=False, header=False)

print('Terminating')
api.terminate()
