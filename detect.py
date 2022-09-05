from asyncore import read
from unittest.mock import CallableMixin
import cv2
import numpy
import dlib

camera =cv2.VideoCapture(0)
face_detector =dlib.get_frontal_face_detector()
facial_points=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    obj,get_web_cam_data=camera.read()
    grayscale=cv2.cvtColor(get_web_cam_data,cv2.COLOR_BGR2GRAY)
    face=face_detector(grayscale)
    
    for i in face:
        points=facial_points(grayscale,i)
        left_eye_brow=(points.part(21).x,points.part(21).y)
        right_eye_brow=(points.part(22).x,points.part(22).y) 
        mid_x=(left_eye_brow[0]+right_eye_brow[0])//2
        mid_y=(left_eye_brow[1]+right_eye_brow[1])//2
        print("midpoint",mid_x,mid_y)
        print("coordinates",left_eye_brow,right_eye_brow)
        cv2.circle(get_web_cam_data,(mid_x,mid_y),8,(0,0,255),-1) 
        cv2.putText(get_web_cam_data,"Sundari oru pottu",(10,500),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1,cv2.LINE_AA)
    cv2.imshow("Sundari",get_web_cam_data)
    
    keyboard_breaker=cv2.waitKey(1)
    if  keyboard_breaker==0:
        break
    

