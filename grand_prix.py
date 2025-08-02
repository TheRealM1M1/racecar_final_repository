"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: template.py << [Modify with your own file name!]

Title: [PLACEHOLDER] << [Modify with your own title]

Author: [PLACEHOLDER] << [Write your name or team name here]

Purpose: [PLACEHOLDER] << [Write the purpose of the script here]

Expected Outcome: [PLACEHOLDER] << [Write what you expect will happen when you run
the script.]
"""

########################################################################################
# Imports
########################################################################################

import sys

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(0, '../library')
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################
rc = racecar_core.create_racecar()

MIN_CONTOUR_AREA = 1000


CROP_FLOOR = ((280, 0), (340, rc.camera.get_width())) # set the crop for the image 

# TODO Part 1: Determine the HSV color threshold pairs for GREEN and RED
# Colors, stored as a pair (hsv_min, hsv_max) Hint: Lab E!
BLUE = ((85, 100, 100), (130, 255, 255))  # The HSV range for the color blue
GREEN = ((30, 77, 186), (166, 170, 207))  # GREEN FOR REAL LIFE
#GREEN = ((35,50,0), (80,255,255)) # GREEN FOR SIM
RED = ((0,50,50),(10,255,255))  # The HSV range for the color red
ORANGE = ((9, 86, 170), (36, 255, 255))

COLORS = [("ORANGE", ORANGE[0], ORANGE[1]), ("GREEN", GREEN[0], GREEN[1])]



# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour
previous_error = 0
previous_center = None


# Declare any global variables here
speed = 0
angle = 0
last_marker_id = None


########################################################################################
# Functions
########################################################################################
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
    global angle, speed

    scan = rc.lidar.get_samples()
    sweep_range_deg = 120         # Total sweep range (-60° to +60°)
    triangle_span_deg = 30        # Width of each triangle
    triangle_depth = 100          # How far out to check (meters)
    min_clearance = 150           # Required clearance inside triangle
    angle_step = 2    

    best_direction = 0
    max_clearance = 0

    required_clearance = 120

    for center_angle in range(left_bound, right_bound, angle_step):
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

    angle = best_direction / p
    speed = 0.6 if abs(best_direction) > 45 else 0.6
    speed = rc_utils.clamp(speed, -1.0, 1.0)
    angle = rc_utils.clamp(angle, -1.0, 1.0)

def ar_marker():
    """
    Define function for extracting AR marker ids
    """
    image = rc.camera.get_color_image() # find the image from color camera
    markers = rc_utils.get_ar_markers(image) # use racecar function to find AR markers in the current frame

    # if we see any AR markers, return the ID of the first one
    if markers: 
        return markers[0].get_id()
    # if no markers then return None
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

        kp = abs(error) * -0.000008 # create a reactive kp that depends on the magnitude of the error


        # kd = 0.00012
        # d = (error - previous_error)/rc.get_delta_time()

        angle = kp * error # calculate angle by mutliplying the calculated kp by the error

        speed = max( 0.5, 1 - abs(angle)) # set the speed using a speed controller based on the angle that we are turning

        angle = rc_utils.clamp(angle, -1, 1) # clamp the angle
        previous_error = error # set previous error as the error of this loop (only used for kd when its implemented)

def cone_slalom():
    global contour_center
    global contour_area
    global contour_color

    # insert cone slalom code 



        

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed, angle
    
    # initialize speed and angle at the start of program
    speed = 0
    angle = 0

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    # declare global variables
    global speed, angle, last_marker_id

    # look for AR markers in every frame and store the latest ID
    current_id = ar_marker()
    if current_id is not None:
        last_marker_id == current_id
    # print the last marker ID (mainly for debugging)
    print(last_marker_id)

    # run wall follower/path planner for the first part of the grand prix (id = 1)    
    if last_marker_id == 1:
        path_planner(-30, 31, 70)
    # Run cone slalom code for the second part of the course (id = 2)
    if last_marker_id == 2:
        cone_slalom()
    # Run combination of line follow and path planner for the third part of the course (id = 3; gates and line follow)
    if last_marker_id == 3:
        line_follow((GREEN, BLUE))
        if contour_center is None:
            path_planner(-60, 61, 70)
            scan = rc_utils.get_lidar_scan()
            if rc_utils.get_lidar_average_distance(0, 30) < 200:
                speed = 0
    # Run path planner (or potentially line follower) for the circular part of the course
    if last_marker_id == 4:
        path_planner(-45, 16, 70)
        # line_follow(GREEN, BLUE)
    

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
