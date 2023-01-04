import numpy as np
import camera_tools as ct
import cv2
import datetime
from FableAPI.fable_init import api
import csv
import pickle
from utils import initialize_camera, initialize_robot


cam = ct.prepare_camera()
print(cam.isOpened())
print(cam.read())

i = 0

initialize_camera(cam)
module = initialize_robot()

# Write DATA COLLECTION part - the following is a dummy code
# Remember to check the battery level and to calibrate the camera
# Some important steps are: 
# 1. Define an input/workspace for the robot; 
# 2. Collect robot data and target data
# 3. Save the data needed for the training

n_t1 = 10
n_t2 = 10

t1 = np.tile(np.linspace(-86, 0, n_t1), n_t2)  # repeat the vector
t2 = np.repeat(np.linspace(-86, 86, n_t2), n_t1)  # repeat each element
thetas = np.stack((t1, t2))

num_datapoints = n_t1 * n_t2
  
api.setPos(thetas[0, i], thetas[1, i], module)


class TestClass:
    def __init__(self, num_datapoints):
        self.i = 0
        self.num_datapoints = num_datapoints
        self.data = np.zeros((num_datapoints, 4))
        self.time_of_move = datetime.datetime.now()

        with open('robot_pos.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["robot_pos_x", "robot_pos_y", "img_pos_x", "img_pos_y"])

    def go(self):
        if self.i >= num_datapoints:
            return True
        
        img = ct.capture_image(cam)
        x, y = ct.locate(img)
        if (datetime.datetime.now() - self.time_of_move).total_seconds() > 0.5:
            if x is not None:
                print(x, y)
                tmeas1 = api.getPos(0, module)
                tmeas2 = api.getPos(1, module)
                self.data[self.i, :] = np.array([tmeas1, tmeas2, x, y])
                self.i += 1

                # set new pos
                if self.i != num_datapoints:
                    api.setPos(thetas[0, self.i], thetas[1, self.i], module)
                    self.time_of_move = datetime.datetime.now()
            else:
                print("Obj not found")
        
        return False
    
    '''def write_csv(self):
        with open('robot_pos.csv', 'a', newline='') as self.file:
            self.writer.writerows(self.data)'''


test = TestClass(num_datapoints)

while True:
    if test.go():
        break

print('Terminating')
api.terminate()

# DONE SAVE .csv file with robot pos data and target location x,y

with open('robot_pos.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(test.data)
print("file written")

with open("training_data.p", 'wb') as f:
    pickle.dump(test.data, f)
