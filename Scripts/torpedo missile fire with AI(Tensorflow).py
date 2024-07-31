
import numpy as np
import argparse
import tensorflow as tf
import cv2
import math
import time
import serial
from pymavlink import mavutil
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Patch for TensorFlow 1.x compatibility
utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile

def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


def calculate_motor_speeds(distance_cm):
    min_speed = -1000
    max_speed = 1000

    # Adjust speeds based on the distance between red and blue marks
    if distance_cm <= 50:
        reduction_factor = (distance_cm - 10) / (50 - 10)
        speed = min_speed + reduction_factor * (max_speed - min_speed)
    else:
        speed = max_speed

    return speed

def calculate_motor_speeds2(distance_cm):
    min_speed = 500
    max_speed = 1000

    # Adjust speeds based on the distance between red and blue marks
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
    # Convert float values to integers if necessary
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




def run_inference(model, category_index, cap, pixel_to_cm, master):
    while True:
        ret, image = cap.read()
        cell_phone_centers = []
        image = np.asarray(image)
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis,...]
        detections = model(input_tensor)

        detection_classes = detections['detection_classes'][0].numpy().astype(int)
        detection_scores = detections['detection_scores'][0].numpy()
        detection_boxes = detections['detection_boxes'][0].numpy()

        for i in range(len(detection_classes)):
            if detection_classes[i] == 1 and detection_scores[i] > 0.5: 
                ymin, xmin, ymax, xmax = detection_boxes[i]
                height, width, _ = image.shape
                center_x = int((xmin + xmax) / 2 * width)
                center_y = int((ymin + ymax) / 2 * height)
                apparent_height_px = (ymax - ymin) * image.shape[0]
                iphone_height_mm = 146.7  # Height of iPhone 12 Pro in millimeters
                distance_cm1 = (focal_length * iphone_height_mm) / apparent_height_px
                cell_phone_centers.append((center_x, center_y))
                print("CenterX: ", center_x, "CenterY: ", center_y)

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
            print(blue_center[1])
            motor_speeds_1_to_4 = calculate_motor_speeds2(distance_1_to_4_cm)
            motor_speeds_5_to_6 = calculate_motor_speeds(distance_5_to_6_cm)
            motor_speeds_7_to_8 = calculate_motor_speeds(distance_7_to_8_cm)

            if distance_5_to_6_cm < 10 and distance_7_to_8_cm < 10 and distance_1_to_4_cm < 10:
                min_speed = 0
                max_speed = 1000

                if distance_cm1 >= 100:
                    send_manual_control(master, 1000, 0, 0, 0)
                    print("100") 

                elif distance_cm1 >= 30 and distance_cm1 <= 100:
                    forward = min_speed + ((distance_cm1 - 30) / (100 - 30)) * (max_speed - min_speed)
                    send_manual_control(master, forward, 0, 0, 0)
                    print("30-100")

                elif distance_cm1 <= 30:
                    send_manual_control(master, 0, 0, 0, 0)
                    ser.write(b'b')  # send 'b' to Arduino
                    print("Sent 'b' to Arduino")
                




                
                #print("Motor 1and4:", motor_speeds_1_to_4)
                #print("Motor 5and6:", motor_speeds_5_to_6)
                #print("Motor 7and8:", motor_speeds_7_to_8)
                #print("distance measurement started")
                #print("distance: ", int(distance_cm1))


                send_manual_control(master, 0, -motor_speeds_5_to_6, 500, 0)

                send_manual_control(master, 0, motor_speeds_7_to_8, 500, 0) 
            

             
                
            print("Motor 1and4:", motor_speeds_1_to_4)
            print("Motor 5and6:", motor_speeds_5_to_6)
            print("Motor 7and8:", motor_speeds_7_to_8)
                
            ser.write(b'k')  # send 'b' to Arduino
            #print("Sent 'k' to Arduino")
            if center_y < 210:
                send_manual_control(master, 0, 0, motor_speeds_1_to_4, 0)
            elif center_y > 270:
                send_manual_control(master, 0, 0, -motor_speeds_1_to_4, 0)

            send_manual_control(master, 0, 0, 500, -motor_speeds_5_to_6)

            send_manual_control(master, 0, 0, 500, motor_speeds_7_to_8) 

           # print("stabil")

        else:
            motor_speeds_1_to_4 = 500
            motor_speeds_5_to_6 = 0
            motor_speeds_7_to_8 = 0
            send_manual_control(master, 0, -motor_speeds_5_to_6, motor_speeds_1_to_4, 0)
            send_manual_control(master, 0, motor_speeds_7_to_8, motor_speeds_1_to_4, 0) 
            

        cv2.imshow('object detection', cv2.resize(image, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects inside webcam videostream')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    
    args = parser.parse_args()

    known_distance = 30  # in centimeters
    focal_length = 84.02431481762936
    pixel_to_cm = known_distance / focal_length

    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)

    cap = cv2.VideoCapture(1)
    master = mavutil.mavlink_connection("COM8", baud=115200)
    master.wait_heartbeat()
   

    ser = serial.Serial('COM9', 9600)

    # Arm the Pixhawk
    master.arducopter_arm()
    time.sleep(2)  


    # Run inference and send motor speeds to Pixhawk
    run_inference(detection_model, category_index, cap, pixel_to_cm, master)

    # Close the connection
    master.close()
    ser.close() 
