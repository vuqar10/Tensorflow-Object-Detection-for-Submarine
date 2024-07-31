Tensorflow Object Detection for Submarine

This project implements real-time object detection for a submarine using TensorFlow models. The system is fully AI-operated and eco-friendly, emphasizing vector degrees of freedom (dof) and powered by LiPo batteries. It includes integration with Pixhawk, bidirectional ESC, and brushless motors.

Introduction

This project leverages TensorFlow and OpenCV for object detection, with a specific focus on detecting targets in underwater environments. The detected objects are used to adjust the submarine's motor speeds for navigation.

Features

 1. Real-time object detection using TensorFlow Lite.
 2. Integration with Pixhawk using MAVLink for communication.
 3. Control of bidirectional ESC and brushless motors via Arduino.
 4. Detection and navigation based on specific target colors (red, blue, green).
    
Requirements

 1. TensorFlow 2.x
 2. OpenCV
 3. MAVLink
 4. Pixhawk
 5. Arduino

For using Tensorflow 2 Object Detection You have to install Tensorflow Object Detection API to Your Decvice. And for using Tensorflow Lite, just "pip install tflite-runtime".
