import camera_tools as ct
from FableAPI.fable_init import api


# Initialization of the camera. Wait for sensible stuff
def initialize_camera(cam):
    while True:
        frame = ct.capture_image(cam)

        x, _ = ct.locate(frame)

        if x is not None:
            break


# Initialize the robot module
def initialize_robot(module=None):
    api.setup(blocking=True)
    # Find all the robots and return their IDs
    print('Search for modules')
    moduleids = api.discoverModules()

    if module is None:
        module = moduleids[0]
    print('Found modules: ', moduleids)
    api.setPos(40, 40, module)
    api.sleep(0.5)
    print(api.getPos('X', module))
    print(api.getPos('Y', module))
    return module
