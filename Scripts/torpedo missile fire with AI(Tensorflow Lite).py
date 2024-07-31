import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import math
from threading import Thread
import importlib.util

import serial
from pymavlink import mavutil



master = mavutil.mavlink_connection("COM8", baud=115200)
master.wait_heartbeat()
   

ser = serial.Serial('COM9', 9600)


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
class VideoStream:
    def __init__(self, resolution=(800,600), framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Motor control functions
def calculate_motor_speeds(distance_cm):
    min_speed = -1000
    max_speed = 1000
    if distance_cm <= 50:
        reduction_factor = (distance_cm - 10) / (50 - 10)
        speed = min_speed + reduction_factor * (max_speed - min_speed)
    else:
        speed = max_speed
    return speed

def calculate_motor_speeds2(distance_cm):
    min_speed = 500
    max_speed = 1000
    if distance_cm <= 50:
        reduction_factor = (distance_cm - 10) / (50 - 10)
        speed = min_speed + reduction_factor * (max_speed - min_speed)
    else:
        speed = max_speed
    return speed

def calculate_distance(red_center, blue_center, pixel_to_cm):
    dx = blue_center[0] - red_center[0]
    dy = blue_center[1] - red_center[1]
    distance_pixel = math.sqrt(dx**2 + dy**2)
    return distance_pixel * pixel_to_cm

def send_manual_control(master, x, y, z, r):
    x = int(x)
    y = int(y)
    z = int(z)
    r = int(r)
    master.mav.manual_control_send(
        master.target_system,
        x,
        y,
        z,
        r,
        0  # Assuming buttons are not used here
    )

# Main program logic
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.', default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu


known_distance = 30  # in centimeters
focal_length = 84.02431481762936
pixel_to_cm = known_distance / focal_length


# Import TensorFlow libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    if GRAPH_NAME == 'detect.tflite':
        GRAPH_NAME = 'edgetpu.tflite'

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del(labels[0])

# Load the TensorFlow Lite model.
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5
outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)

while True:
    frame1 = videostream.read()
    image = np.asarray(frame1)

    # Get the expected input dimensions
    input_shape = input_details[0]['shape']
    target_height, target_width = input_shape[1], input_shape[2]

    # Resize the input image to match the expected input shape of the model
    image_resized = cv2.resize(image, (target_width, target_height))

    # Prepare input tensor
    if floating_model:
        input_data = np.float32(image_resized) / 255.0
    else:
        input_data = np.uint8(image_resized)

    input_data = np.expand_dims(input_data, axis=0)

    # Set the input tensor for the interpreter
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Perform inference
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    cell_phone_centers = []
    for i in range(len(classes)):
        if classes[i] == 1 and scores[i] > min_conf_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            height, width, _ = image.shape
            center_x = int((xmin + xmax) / 2 * width)
            center_y = int((ymin + ymax) / 2 * height)
            apparent_height_px = (ymax - ymin) * image.shape[0]
            iphone_height_mm = 146.7
            distance_cm1 = (focal_length * iphone_height_mm) / apparent_height_px
            cell_phone_centers.append((center_x, center_y))

    window_center = (image.shape[1] // 2, image.shape[0] // 2)

    for center in cell_phone_centers:
        cv2.circle(image, center, 5, (255, 0, 0), -1)
    cv2.circle(image, window_center, 5, (0, 0, 255), -1)

    if len(cell_phone_centers) > 0:
        red_center = cell_phone_centers[0]
        blue_center = window_center
        distance_1_to_4_cm = abs(red_center[1] - blue_center[1])
        distance_5_to_6_cm = calculate_distance(red_center, blue_center, pixel_to_cm) if red_center[0] < blue_center[0] else 0
        distance_7_to_8_cm = calculate_distance(red_center, blue_center, pixel_to_cm) if red_center[0] > blue_center[0] else 0

        motor_speeds_1_to_4 = calculate_motor_speeds2(distance_1_to_4_cm)
        motor_speeds_5_to_6 = calculate_motor_speeds(distance_5_to_6_cm)
        motor_speeds_7_to_8 = calculate_motor_speeds(distance_7_to_8_cm)

        if distance_5_to_6_cm < 10 and distance_7_to_8_cm < 10 and distance_1_to_4_cm < 10:
            min_speed = 0
            max_speed = 1000

            if distance_cm1 >= 100:
                print("100")
                send_manual_control(master, 1000, 0, 0, 0)

            elif 30 <= distance_cm1 <= 100:
                forward = min_speed + ((distance_cm1 - 30) / (100 - 30)) * (max_speed - min_speed)
                send_manual_control(master, forward, 0, 0, 0)
                print("30-100")

            elif distance_cm1 <= 30:
                send_manual_control(master, 0, 0, 0, 0)
                ser.write(b'b')
                print("arduino")
            print(distance_cm1)
        send_manual_control(master, 0, -motor_speeds_5_to_6, 500, 0)
        send_manual_control(master, 0, motor_speeds_7_to_8, 500, 0) 

        if center_y < 210:
            send_manual_control(master, 0, 0, motor_speeds_1_to_4, 0)
            print("down")
        elif center_y > 270:
            send_manual_control(master, 0, 0, -motor_speeds_1_to_4, 0)
            print("up")

        send_manual_control(master, 0, 0, 500, -motor_speeds_5_to_6)
        send_manual_control(master, 0, 0, 500, motor_speeds_7_to_8) 

    else:
        motor_speeds_1_to_4 = 500
        motor_speeds_5_to_6 = 0
        motor_speeds_7_to_8 = 0
        send_manual_control(master, 0, -motor_speeds_5_to_6, motor_speeds_1_to_4, 0)
        send_manual_control(master, 0, motor_speeds_7_to_8, motor_speeds_1_to_4, 0) 

    cv2.imshow('object detection', cv2.resize(image, (800, 600)))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        videostream.stop()
        master.close()
        ser.close() 
        cv2.destroyAllWindows()
        break