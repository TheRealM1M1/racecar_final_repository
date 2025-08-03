"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: conga.py

Title: Conga Line

Purpose: Autonomously follow the car in front by using a object detection model
"""

########################################################################################
# Imports
########################################################################################

import sys

sys.path.insert(0, '../library')
import racecar_core

import cv2

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

import racecar_utils as rc_utils

# Define paths to model and label directories
# TODO: change this stuff to match our file structure
default_path = 'models_new' # location of model weights and labels
model_name = 'car_follow_edgetpu.tflite'
label_name = 'labels.txt'

model_path = default_path + "/" + model_name
label_path = default_path + "/" + label_name

# Define thresholds and number of classes to output
SCORE_THRESH = 0.1
NUM_CLASSES = 3
########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here
speed = 0
angle = 0
last_angle = 0

########################################################################################
# Functions
########################################################################################
print('Loading {} with {} labels.'.format(model_path, label_path))
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()
labels = read_label_file(label_path)
inference_size = input_size(interpreter)

def get_obj_and_type(cv2_im, inference_size, objs):
    height, width, _ = cv2_im.shape
    max_score = 0
    correct_obj = None
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        if obj.score > max_score:
            max_score = obj.score
            correct_obj = obj
    if (correct_obj is not None):
        bbox = correct_obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)
        center = ((x0+x1)/2, (y0+y1)/2)
        id = correct_obj.id
    return center, id, correct_obj


# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed
    global angle 
    
    angle = 0
    last_angle = 0

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global speed, angle, last_angle
    image = rc.camera.get_color_image()
    if image is not None:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, inference_size)
        run_inference(interpreter, rgb_image.tobytes())
        objs = get_objects(interpreter, SCORE_THRESH)[:NUM_CLASSES]    
        if (len(objs) != 0):
            center, id, obj = get_obj_and_type(image, inference_size, objs)
            pv_a = center[0]
            setpoint_a = rc.camera.get_width() / 2
            error_a = setpoint_a - pv_a
            print(f"{error_a} = {setpoint_a} - {pv_a}")
            kp_a = -0.0015
            angle = rc_utils.clamp(kp_a * error_a, -1, 1)
            print(center)
            pv_s = obj.bbox.area
            setpoint_s = 18000
            error_s = setpoint_s - pv_s
            kp_s = 0.00004
            
            speed = rc_utils.clamp(kp_s * error_s, 0.3, 1)
            last_angle = angle
        else:
            speed = 0.4
            angle = 0
    else:
        print("image is none")
        speed = 0
        angle = 0
    print(f"speed: {speed} | angle: {angle}")
    rc.drive.set_speed_angle(speed, angle)

# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    pass # Remove 'pass and write your source code for the update_slow() function here


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()

