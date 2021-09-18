# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 20:30:08 2021

@author: frang
"""
import pickle
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import pandas as pd
import functions as fun
import sys
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    # parser.add_argument('--upper_body_only', action='store_true')  # 0.8.3 or less
    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='face mesh min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='face mesh min_tracking_confidence',
                        type=int,
                        default=0.5)

    # Threshold for fWLR distance to detect new person in front of camera (default 0.1)
    parser.add_argument("--threshold", help='face similarity threshold', type=float, default=0.1)
    args = parser.parse_args()

    return args

def main():
    # Read arguments
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    threshold = args.threshold

    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    
    # Init camera##############################################################
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Init mediapipe holistic##################################################
    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions
    holistic = mp_holistic.Holistic(
        # upper_body_only=upper_body_only,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Calculate FPS ###########################################################
    cvFpsCalc = fun.CvFpsCalc(buffer_len=10)

    # Vars to count and detect new person in front of camera ##################
    firstPerson = True
    currentPerson = None
    newPerson = None
    countPeople = 0
    people = []

    # OpenCV video stream
    while cap.isOpened():
        # Calculate FPS
        display_fps = cvFpsCalc.get()
        # Capture image
        ret, frame = cap.read()
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        # Make Detections with Mediapipe
        results = holistic.process(image)
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        originalimage = image.copy()

        # Face detected by MediaPipe
        if results.face_landmarks:
            # Calculate features (width Ratio and length Ratio)
            x, y, xw, yh = fun.calc_bounding_rect(image, results.face_landmarks)
            faceWRatio, faceLRatio = fun.calculateFaceRatios(results)
            
            # Create temporal person object
            currentPerson = fun.Person(
                0,
                originalimage, #image[y:yh,x:xw],
                faceWRatio,
                faceLRatio
            )
            # Same person similarity
            samePerson, distance = fun.isSamePerson(currentPerson, newPerson, threshold, "Custom")

            if firstPerson or not samePerson:
                # New person detected
                if firstPerson:
                    firstPerson = False
                countPeople += 1
                newPerson = fun.Person(
                    countPeople,
                    currentPerson.image,
                    currentPerson.faceWRatio,
                    currentPerson.faceLRatio
                )
                # Add new person to array for post-processing
                people.append(newPerson)

            # General display
            fun.writeInfo(image, display_fps, countPeople, distance)
        
        key = cv2.waitKey(1)
        if key == 27:
            break

        # Draw landmarks
        image = fun.drawLandmarks(image, results)
        cv2.imshow('Raw Webcam Feed', image)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()