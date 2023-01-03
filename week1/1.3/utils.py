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
    module_ids = api.discoverModules()

    if module is None:
        module = module_ids[0]
    print('Found modules: ', module_ids)
    api.setPos(0, 0, module)
    api.sleep(0.5)
    return module
