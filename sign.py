"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: sign.py

Title: Sign Detection

Author: Ferrari Rochers/Team 6
        Mihir Tare, Vyom Siriyapu, Adhyayan Gupta, Andrew Sperry

Purpose: Detect different traffic signs while wall following and respond appropriately
"""

########################################################################################
# Imports
########################################################################################

import sys

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(0, '../library')
import racecar_core
import cv2
import os
import time
import racecar_utils as rc_utils

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

# Define paths to model and label directories
# TODO: change this stuff to match our file structure
default_path = 'models_new' # location of model weights and labels
model_name = 'traffic_v2_edgetpu.tflite'
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
prvious_id = None

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
    return center, id, correct_obj, max_score

def calc_angle(angle):
    if angle < 0:
        return 360 + angle
    return angle
# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global previous_id

    previous_id = 0

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global speed, angle, last_angle, previous_id
    speed = 0.55

    scan = rc.lidar.get_samples()

    
    sweep_range_deg = 120         # Total sweep range (-60° to +60°)
    triangle_span_deg = 30        # Width of each triangle
    triangle_depth = 100          # How far out to check (meters)
    min_clearance = 150           # Required clearance inside triangle
    angle_step = 2    

    best_direction = 0
    max_clearance = 0

    required_clearance = 120

    for center_angle in range(-75, 76, angle_step):
        start_angle = center_angle - triangle_span_deg // 2
        end_angle = center_angle + triangle_span_deg // 2
        

        # Sample LiDAR points inside triangle
        inside = []
        for angle in range(start_angle, end_angle + 1):
            dist = rc_utils.get_lidar_average_distance(scan, calc_angle(angle))
            if dist > triangle_depth:
                inside.append(dist)

        # Check clearance
        if len(inside) == 0 or min(inside) < required_clearance:
            continue
        min_dist = min(inside)
        
        if min_dist > max_clearance:
            best_direction = center_angle
            max_clearance = min_dist

    
    steering = best_direction / 70.0
    image = rc.camera.get_color_image()
    if image is not None:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, inference_size)
        run_inference(interpreter, rgb_image.tobytes())
        objs = get_objects(interpreter, SCORE_THRESH)[:NUM_CLASSES]    
        if (len(objs) != 0):
            center, id, obj, score = get_obj_and_type(image, inference_size, objs)
            if id == 0 and score > 0.30:
                print("STOP")
                speed = 0
            elif id == 4:
                print("YIELD")
                speed = 0.4  
            elif id == 1:
                print("Go around")
                steering = 0.31
            
            #speed = rc_utils.clamp(kp_s * error_s, 0.3, 1)
            last_angle = angle
            previous_id = id
        else:
            speed = 0.55
            angle = 0
    else:
        print("image is none")
        speed = 0
        angle = 0
        
    
    #if abs(best_direction) > 40: speed = 0.6
    steering = rc_utils.clamp(steering, -1.0, 1.0)
    print(speed)
    rc.drive.set_speed_angle(speed, steering)
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

