import numpy as np
import camera_tools as ct
import cv2
import datetime
from FableAPI.fable_init import api

cam = ct.prepare_camera()
print(cam.isOpened())
print(cam.read())

i = 0

# Initialization of the camera. Wait for sensible stuff
def initialize_camera(cam):
    while True:
        frame = ct.capture_image(cam)

        x, _ = ct.locate(frame)

        if x is not None:
            break     

def initialize_robot(module=None):
    api.setup(blocking=True)
    # Find all the robots and return their IDs
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

# Write DATA COLLECTION part - the following is a dummy code
# Remember to check the battery level and to calibrate the camera
# Some important steps are: 
# 1. Define an input/workspace for the robot; 
# 2. Collect robot data and target data
# 3. Save the data needed for the training

n_t1 = 10
n_t2 = 10

t1 = np.tile(np.linspace(-85, 86, n_t1), n_t2) # repeat the vector
t2 = np.repeat(np.linspace(0, 86, n_t2), n_t1) # repeat each element
thetas = np.stack((t1,t2))

num_datapoints = n_t1*n_t2

api.setPos(thetas[0,i], thetas[1,i], module)

class TestClass:
    def __init__(self, num_datapoints):
        self.i = 0
        self.num_datapoints = num_datapoints
        self.data = np.zeros( (num_datapoints, 4) )
        self.time_of_move = datetime.datetime.now()

    def go(self):
        if self.i >= num_datapoints:
            return True
        
        img = ct.capture_image(cam)
        x, y = ct.locate(img)
        if (datetime.datetime.now() - self.time_of_move).total_seconds() > 2.0:
            if x is not None:
                print(x, y)
                tmeas1 = api.getPos(0,module)
                tmeas2 = api.getPos(1,module)
                self.data[i,:] = np.array([tmeas1, tmeas2, x, y])
                self.i += 1

                # set new pos
                if self.i != num_datapoints:
                    api.setPos(thetas[0,self.i], thetas[1,self.i], module)
                    self.time_of_move = datetime.datetime.now()
            else:
                print("Obj not found")
                
        return False

test = TestClass(num_datapoints)
test.go()    


print('Terminating')
api.terminate()

# TODO SAVE .csv file with robot pos data and target location x,y

