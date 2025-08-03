"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo

File Name: grand_prix.py

Title: Grand Prix Code

Author: Ferrari Rochers - Mihir Tare, Vyom Siriyapu, Andrew Sperry, Adhyayan (Adi) Gupta

Purpose: Compile the work of our 4 weeks at BWSI into one final code to run through the 2025 summer Grand Prix.

Expected Outcome: Win!!!
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, '../library')
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################
rc = racecar_core.create_racecar()

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

MIN_CONTOUR_AREA = 1000


CROP_FLOOR = ((280, 0), (340, rc.camera.get_width())) # set the crop for the image 

# TODO Part 1: Determine the HSV color threshold pairs for GREEN and RED
# Colors, stored as a pair (hsv_min, hsv_max) Hint: Lab E!
BLUE = ((85, 100, 100), (130, 255, 255))  # The HSV range for the color blue
GREEN = ((30, 77, 186), (166, 170, 207))  # GREEN FOR REAL LIFE
GREEN_CONE = ((40, 109, 1), (74, 255, 255))
#GREEN = ((35,50,0), (80,255,255)) # GREEN FOR SIM
RED = ((0,50,50),(10,255,255))  # The HSV range for the color red
ORANGE = ((9, 86, 170), (36, 255, 255))
ORANGE_CONE = ((1, 80, 181), (20, 255, 255))
COLORS = [("ORANGE", ORANGE[0], ORANGE[1]), ("GREEN", GREEN[0], GREEN[1])]
CONE_COLORS = [("GREEN", GREEN_CONE[0], GREEN_CONE[1]), ("ORANGE", ORANGE_CONE[0], ORANGE_CONE[1])]

class ARMarker:
    def __init__(self, marker_id, marker_corners, orientation, area):
        self.id = marker_id
        self.corners = marker_corners
        self.orientation = orientation # Orientation of the marker
        self.area = area # Area of the marker
        self.color = ""
        self.color_area = 0



# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour
previous_error = 0
previous_center = None
previous_color = None


# Declare any global variables here
speed = 0
angle = 0
last_marker_id = None

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference


# Define paths to model and label directories
# TODO: change this stuff to match our file structure
default_path = 'models_new' # location of model weights and labels
model_name = 'dynamic_gate_edgetpu.tflite'
label_name = 'labels.txt'

model_path = default_path + "/" + model_name
label_path = default_path + "/" + label_name

# Define thresholds and number of classes to output
SCORE_THRESH = 0.1
NUM_CLASSES = 3
########################################################################################
# Functions
########################################################################################
def detect_AR_Tag(image):
    markers = []

    corners, ids, _ = detector.detectMarkers(image)

    for i in range(len(corners)):
        current_corners = corners[i][0]

        if current_corners[0][0] == current_corners[1][0]:
            if current_corners[0][1] > current_corners[1][1]:
                orientation = "LEFT"
            else:
                    orientation = "RIGHT"
        else:
            if current_corners[0][0] > current_corners[1][0]:
                orientation = "DOWN"
            else:
                orientation = "UP"

        area = abs((current_corners[2][0] - current_corners[0][0]) * (current_corners[2][1] - current_corners[0][1]))


        marker = ARMarker(ids[i][0], current_corners, orientation, area)
        if marker.area > 00: markers.append(marker)
    
    #cv2.aruco.drawDetectedMarkers(image, corners, ids, (0, 255, 0))

    return markers, image

def update_contour(COLOR_PRIORITY):
    """
    Define function for extracting color contours based on the color priority
    Params: color priority of what colors to look for first
    """
    # globalize variables
    global contour_center
    global contour_area

    image = rc.camera.get_color_image() # collect color image

    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # Crop the image to the floor directly in front of the car
        contour_center = None
        image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])
       
        contours = rc_utils.find_contours(image, COLOR_PRIORITY[0][0], COLOR_PRIORITY[0][1]) # find contours for the prioritized color
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA) # pick the largest contour
        if contour is not None and rc_utils.get_contour_area(contour) > MIN_CONTOUR_AREA: # check if contour exists and is bigger than the minimum area
            contour_center = rc_utils.get_contour_center(contour) # set the contour to the biggest one we find
            contour_area = rc_utils.get_contour_area(contour)
        else: # repeat for the second color in priority list
            contours = rc_utils.find_contours(image, COLOR_PRIORITY[1][0], COLOR_PRIORITY[1][1])
            contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
            if contour is not None:
                contour_center = rc_utils.get_contour_center(contour)
                contour_area = rc_utils.get_contour_area(contour)
        

def update_contour_cone():
    global contour_center
    global contour_area
    global contour_color

    image = rc.camera.get_color_image() # get color image

    if image is None:
        contour_center = None
        contour_area = 0
        contour_color = None
    else:
        # Crop the image to the floor directly in front of the car
        image = rc_utils.crop(image, (180,0), (rc.camera.get_height(), rc.camera.get_width()))
        contour_area = 0
        contour_color = None
        # TODO Part 2: Search for line colors, and update the global variables
        # contour_center and contour_area with the largest contour found
        for color in COLORS: # loop through the colors in the list
            contours = rc_utils.find_contours(image, color[1], color[2]) # find contours within the HSV range
            contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA) # find the largest contour from that list
            if contour is not None and rc_utils.get_contour_area(contour) > contour_area: # if the contour exist and it's big enough, then set that as the contour and the color
                contour_center = rc_utils.get_contour_center(contour)
                contour_area = rc_utils.get_contour_area(contour)
                contour_color = color[0]

def calc_angle(angle): 
    """
    Define function for making negative angles correct for the car
    """
    if angle < 0:
        return 360 + angle
    return angle

def path_planner(left_bound, right_bound, p):
    """
    Define function for our wall follower.
    Params: Left bound and right bound of angles to search through as well as coefficient that converts algorithm to racecar angle
            also the speeds to drive at
    """
    global angle, speed

    scan = rc.lidar.get_samples() # collect LIDAR samples from racecar
    triangle_span_deg = 30        # Width of each triangle
    triangle_depth = 100          # How far out to check (meters)
    angle_step = 2    

    best_direction = 0
    max_clearance = 0

    required_clearance = 120

    # loop through each angle for the bounds given in the paramters
    for center_angle in range(left_bound, right_bound, angle_step):
        # find the start angle and end angle for each individual triangle
        start_angle = center_angle - triangle_span_deg // 2
        end_angle = center_angle + triangle_span_deg // 2
        

        # Sample LiDAR points inside triangle
        inside = []
        for angle in range(start_angle, end_angle + 1):
            dist = rc_utils.get_lidar_average_distance(scan, calc_angle(angle)) # find the average distance for each angle
            if dist > triangle_depth: 
                inside.append(dist) # append the distances that are greater than the depth we are looking for
    

        # Check clearance
        if len(inside) == 0 or min(inside) < required_clearance:
            # ignore the angle if the smallest clearance in the triangle is less that required clearance
            continue
        min_dist = min(inside) 
        
        # determine if this is the most reliable angle if it has the maximum minimum clearance
        if min_dist > max_clearance:
            best_direction = center_angle
            max_clearance = min_dist
    # calculate ouput angle based on the best direction determined and multiplying it (or dividing)by the coefficient in the parameters
    angle = int(best_direction) / 70
    # set speed based on the best direciton
    speed = max(0.5, 0.9 - (abs(best_direction) / 125))

    #clamp speed and angle
    speed = rc_utils.clamp(speed, -1.0, 1.0)
    angle = rc_utils.clamp(angle, -1.0, 1.0)

def ar_marker():
    """
    Define function for extracting AR marker ids
    """
    markers = []
    image = rc.camera.get_color_image() # find the image from color camera
    if image is not None:
        markers = rc_utils.get_ar_markers(image) # use racecar function to find AR markers in the current frame

    # if we see any AR markers, return the ID of the first one
    if markers: 
        return markers[0].get_id()
    # if no markers then return None
    else:
        return None

def line_follow(COLOR_PRIORITY): 
    """
    Define function for a line follower
    Params: color priority for which colored lines to follow (green and blue)
    """
    # global variables
    global speed, angle, previous_error, previous_center

    # Search for contours in the current color image
    update_contour(COLOR_PRIORITY)

    # Choose an angle based on contour_center
    # If we could not find a contour, keep the previous angle
    if contour_center is not None:

        setpoint = 320  # center of screen as setpoint for line follower
        present_value = contour_center[1]  # horizontal position of the line

        error = setpoint - present_value # calculate error between where the line is and where we want it to be

        kp = abs(error) * -0.000012 # create a reactive kp that depends on the magnitude of the error


        # kd = 0.00012
        # d = (error - previous_error)/rc.get_delta_time()

        angle = kp * error # calculate angle by mutliplying the calculated kp by the error

        speed = max( 0.5, 0.8 - abs(angle)) # set the speed using a speed controller based on the angle that we are turning

        angle = rc_utils.clamp(angle, -1, 1) # clamp the angle
        previous_error = error # set previous error as the error of this loop (only used for kd when its implemented)

def cone_slalom():
    global speed
    global angle 
    global previous_color

    speed = 0.55

    scan = rc.lidar.get_samples()
    angle_point, distance_point = rc_utils.get_lidar_closest_point(scan)
    update_contour_cone()

    #print(contour_color, contour_area)
    if contour_color is not None:
        previous_color = contour_color

    if previous_color == "GREEN":
        green(scan)
    elif previous_color == "ORANGE":    
        orange(scan)
    elif previous_color == None: 
        path_planner(-45, 46, 70)
        speed = 0.6

def orange(scan):
    global speed
    global angle

    if contour_color is not None:
        closest_point = rc_utils.get_lidar_closest_point(scan, (300, 60))
        setpoint = rc.camera.get_width() // 2  # center of screen (x = 320)
        present_value = contour_center[1]  # horizontal position of the line

        kp = -0.0325  # proportional coefficient which we calculated

        error = setpoint - present_value
        angle = kp * error

        angle = rc_utils.clamp(angle, -1, 1)

        if closest_point[1] < 110:
            #angle = -0.565
            angle = -0.7
    else:
        closest_point = rc_utils.get_lidar_closest_point(scan, (0, 180))
        if 180 > closest_point[0] > 30:
            #setpoint = 20
            #error = setpoint - closest_point[1]
            #angle = -0.03 * error
            angle = 0.6
            
            angle = rc_utils.clamp(angle, -1, 1)

def green(scan):
    global speed
    global angle

    if contour_color is not None:
        closest_point = rc_utils.get_lidar_closest_point(scan, (300, 60))
        setpoint = rc.camera.get_width() // 2  # center of screen (x = 320)
        present_value = contour_center[1]  # horizontal position of the line

        kp = -0.0325  # proportional coefficient which we calculated

        error = setpoint - present_value
        angle = kp * error

        angle = rc_utils.clamp(angle, -1, 1)

        if closest_point[1] < 100:
            angle = 0.7
    else:
        closest_point = rc_utils.get_lidar_closest_point(scan, (180, 360))
        if 180 < closest_point[0] < 310:
            # setpoint = 20
            # error = setpoint - closest_point[1]
            # angle = 0.03 * error
            angle = -0.6
            angle = rc_utils.clamp(angle, -1, 1)


def dynamic_gate_left():
    global speed, angle, last_angle, center, id, obj, area, interpreter, labels, inference_size
    speed = 0.65
    # while area <= 100:
    #     line_follow()
    path_planner(-60, 10, 70)


def dynamic_gate():
    global speed, angle, last_angle, center, id, obj, area, interpreter, labels, inference_size
    image = rc.camera.get_color_image()
    if image is not None:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, inference_size)
        run_inference(interpreter, rgb_image.tobytes())
        objs = get_objects(interpreter, SCORE_THRESH)[:NUM_CLASSES]  

        if (len(objs) != 0):
            center, id, obj, area = get_obj_and_type(image, inference_size, objs)
            print(id)
            if id == 10:
                dynamic_gate_left()
            elif id == 11:
                pass
                # gate_angles = gate_side_finder(345, 15)
                # drive_thru_gate(gate_angles)

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
        area = (x1-x0)*(y1-y0)
    return center, id, correct_obj, area

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed, angle
    global switch_mode, interpreter, labels, inference_size

    
    # initialize speed and angle at the start of program
    speed = 0
    angle = 0
    
    
    print('Loading {} with {} labels.'.format(model_path, label_path))
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    labels = read_label_file(label_path)
    inference_size = input_size(interpreter)

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    # declare global variables
    global speed, angle, last_marker_id, contour_center

    image = rc.camera.get_color_image()
    markers = []
    # look for AR markers in every frame and store the latest ID
    if image is not None:
        markers, image = detect_AR_Tag(image)
    if markers:
        current_id = markers[0].id
        print(markers[0].area)
    else:
        current_id = None
    
    if current_id is not None:
        last_marker_id = current_id
    print(last_marker_id) 
    # print the last marker ID (mainly for debugging)
    #print(last_marker_id)

    # run wall follower/path planner for the first part of the grand prix (id = 1)    
    if last_marker_id is None:
        path_planner(-30, 31, 70)
    if last_marker_id == 0:
        path_planner(-60, 61, 70)
    if last_marker_id == 1:
        path_planner(-60, 61, 70)
    # Run cone slalom code for the second part of the course (id = 2)
    if last_marker_id == 2:
        path_planner(-60, 61, 70)
    # Run combination of line follow and path planner for the third part of the course (id = 3; gates and line follow)
    if last_marker_id == 3:
        #if area is None or area <= 
        #dynamic_gate()
        path_planner(-30, 31, 70)        
        #if rc_utils.get_lidar_average_distance(scan, 0, 30) < 200:
            #speed = 0
        #else:
            #print("Contour found")
    # Run path planner (or potentially line follower) for the circular part of the course
    if last_marker_id == 4:
        line_follow((BLUE, GREEN))
        if contour_center is None:
            path_planner(-60, 61, 70)
        # line_follow(GREEN, BLUE)
    if last_marker_id == 5:
        path_planner(-60, 60, 70)
    

    # set speed and angle
    rc.drive.set_speed_angle(speed, angle)



# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    pass


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()

