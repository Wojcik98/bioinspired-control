import numpy as np
import torch
import torch_model
import cv2
import camera_tools as ct
from FableAPI.fable_init import api
from utils import initialize_camera, initialize_robot
from time import sleep
import pandas as pd
import pickle
from cmac import CMAC
import pyttsx3
import winsound

cam = ct.prepare_camera()
print(cam.isOpened())  # False
i = 0

initialize_camera(cam)
module = initialize_robot()

# Set move speed
speedX = 80
speedY = 80
api.setSpeed(speedX, speedY, module)

# Set accuracy
accurateX = 'HIGH'
accurateY = 'HIGH'
api.setAccurate(accurateX, accurateY, module)

model = torch_model.Net(4, 50, 2)
model.load_state_dict(torch.load('closed_loop_trained_model.pth'))

use_CMAC = True

n_inputs = 4
n_outputs = 2
n_bases = 10
beta = 2e-2
x_min = [-2] * 2
x_max = [2] * 2


t = [40, 40]

cv2.namedWindow("test")

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

model.eval()

np.random.seed(0)

runs = []

engine.say("Commencing test")
engine.runAndWait()

for i in range(2):
    c = [CMAC(n_bases, x_min, x_max, beta) for _ in range(2)]
    if i>0:
        engine.say(f"Uncreate disturbance")
        engine.runAndWait()
        sleep(2)
        winsound.Beep(1000, 1000)
    engine.say(f"Run {i+1}")
    engine.runAndWait()
    print(f"{i+1}/10")
    run = []
    inp = [np.random.random()*(421-192) + 192, np.random.random()*(333-236)+236]
    imginp = [int(inp[0]), int(inp[1])]
    inp = torch.tensor([inp]).float()
    inp = (inp - 200) / 200
    frame = ct.capture_image(cam)
    xy = ct.locate(frame)
    cv2.circle(frame, imginp, 5, (0, 255, 100), -1)
    cv2.imshow("test", frame)
    k = cv2.waitKey(100)
    if k == 27:
        break
    while xy[0] is None:
        frame = ct.capture_image(cam)
        xy = ct.locate(frame)

        cv2.circle(frame, imginp, 5, (0, 255, 100), -1)

        cv2.imshow("test", frame)
        k = cv2.waitKey(100)
        if k == 27:
            break

        xy = ct.locate(frame)
    current_xy = torch.tensor([xy]).float()
    current_xy = (current_xy - 200) / 200
    run.append(np.linalg.norm(inp - current_xy) * 200 + 200)
    for j in range(25):

        if j == 11:
            engine.say("Create disturbance now")
            engine.runAndWait()
            sleep(2)
            winsound.Beep(1000, 1000)
        else:
            with torch.no_grad():
                m_input = torch.tensor([np.append(inp - current_xy, np.divide(t, 90))]).float()
                outp = model(m_input)
                t = outp.numpy()[0] * 90
            # t = [t[0] + dt[0], t[1] + dt[1]]
            t0 = [api.getPos('X', module), api.getPos('Y', module)]
            k = 0.4

            target = k * (t - t0) + t0

            if use_CMAC:
                c0 = c[0].predict([inp[0][1], current_xy[0][1]])
                print(c0)
                c1 = c[1].predict([inp[0][0], current_xy[0][0]])
                target[0] += c0
                target[1] += c1
                c[0].learn((t - t0)[0])
                c[1].learn((t - t0)[1])

            api.setPos(max(-90, min(90, target[0])), max(-90, min(90, target[1])), module)

            while api.getMoving('X', module) or api.getMoving('Y', module):
                sleep(0.1)

        frame = ct.capture_image(cam)
        xy = ct.locate(frame)
        cv2.circle(frame, imginp, 5, (0, 255, 100), -1)
        cv2.imshow("test", frame)
        k = cv2.waitKey(100)
        if k == 27:
            break
        while xy[0] is None:
            frame = ct.capture_image(cam)
            xy = ct.locate(frame)
            cv2.circle(frame, imginp, 5, (0, 255, 100), -1)
            cv2.imshow("test", frame)
            k = cv2.waitKey(100)
            if k == 27:
                break

            xy = ct.locate(frame)
        current_xy = torch.tensor([xy]).float()
        current_xy = (current_xy - 200) / 200

        run.append(np.linalg.norm(inp - current_xy) * 200 + 200)
        sleep(0.1)
        # sleep(1.5)
    runs.append(run)

engine.say("Experiment concluded")
engine.runAndWait()
print('Collected data:')
print(runs)

df = pd.DataFrame(runs)
df.to_csv('demo_recording.csv', index=False, header=False)

print('Terminating')
api.terminate()
