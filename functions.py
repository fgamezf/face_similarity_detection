# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 23:00:08 2021

@author: frang
"""
import pandas as pd
import cv2 # Import opencv
from collections import deque
import time
import numpy as np
import pickle
import scipy.spatial
import pandas as pd
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

# Simple Person class to store fWLR
class Person(object):
  def __init__(self, id, image, faceWRatio, faceLRatio):
    self.id = id
    self.image=image
    self.faceWRatio = faceWRatio
    self.faceLRatio = faceLRatio  

# Class to calculate FPS
class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv2.getTickCount()
        self._freq = 1000.0 / cv2.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv2.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded

def drawLandmarks(image, results):
    # 1. Draw face landmarks
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
        
    # 2. Right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )

    # 3. Left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
       mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
       mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
       )
    
    return image

def writeInfo(image, display_fps, countPeople, distance):
    # Get font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(image, "FPS:" + str(display_fps), (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.putText(image, 
        f"People count: {countPeople}",
        (10,60), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(image, 
        f"fWLR distance: {distance}",
        (10,90), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return image

def calculate_distance(a,b):
    distanceModule = scipy.spatial.distance
    return distanceModule.euclidean(a, b)

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

# Calculate Face Weight Ratio
def calculateFaceWeightRatio(results):
    
    if results.pose_landmarks is None:
        return 0
    
    # Get coordinates
    rightEyeInner = [
        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE_INNER].x,
        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE_INNER].y
    ]
           
            
    leftEyeInner = [
        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE_INNER].x,
        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE_INNER].y
    ]
            
    rightEar = [
        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x,
        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y
    ]
           
            
    leftEar = [
        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y
    ]
            
    faceWRatio = calculate_distance(leftEyeInner, rightEyeInner) / calculate_distance(leftEar, rightEar)
    return faceWRatio

# Calculate Face Length Ratio
def calculateFaceLengthRatio(results):
    
    if results.face_landmarks is None:
        return 0
    
    # Get coordinates
    nose = [
        results.face_landmarks.landmark[0].x,
        results.face_landmarks.landmark[0].y
    ]
           
            
    top_nose = [
        results.face_landmarks.landmark[167].x,
        results.face_landmarks.landmark[167].y
    ]
            
    chin = [
        results.face_landmarks.landmark[151].x,
        results.face_landmarks.landmark[151].y
    ]
    
            
    faceLRatio = calculate_distance(top_nose, nose) / calculate_distance(nose, chin)
    return faceLRatio

def calculateFaceRatios(results):
    faceWRatio = calculateFaceWeightRatio(results)
    faceLRatio = calculateFaceLengthRatio(results)
    return faceWRatio, faceLRatio

def calculateRatioChanges(personA, personB):
    percentChangeW = get_change(personA.faceWRatio, personB.faceWRatio)
    percentChangeL = get_change(personA.faceLRatio, personB.faceLRatio)
    return percentChangeW, percentChangeL
        
def isSamePerson(personA, personB, threshold, algo):
    samePerson = True
    distance = 0

    if algo is None:
        algo = "Custom"

    if personA is not None and personB is not None:
        if (algo == "Custom"):
            X = [personA.faceWRatio, personA.faceLRatio]
            Y = [personB.faceWRatio, personB.faceLRatio]
            distance = calculate_distance(X, Y)
            # THRESOLD    
            samePerson = (distance < threshold)
        else:
            print ("Face similarity model not found")
    else:
        print ("No data to compare")
    
    return samePerson, distance