#!/usr/bin/env python
from __future__ import print_function 
import roslib
import sys, time
import rospy
import cv2
import message_filters
import numpy as np
from matplotlib import pyplot as plt
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBoxes 

  
bridge = CvBridge()
fgbg = cv2.createBackgroundSubtractorMOG2()
image_pub = rospy.Publisher('ROI_image', Image, queue_size=10)
i=0


class ravev(object):

    def __init__(self, image, x_min , x_max, y_min, y_max, det_class, det_id, var):
        super(ravev, self).__init__()
        self.i=0
        self.obj_type = None



    def recognize(self, cv_image, x_min, x_max, y_min, y_max, det_class, det_id, var):

        roi_bgr=cv_image[y_min:y_max , x_min:x_max]
        fgmask = fgbg.apply(roi_bgr)[1]

        self.i=self.i+1

        # Convert BGR to HSV
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # cv2.namedWindow("ROI_BGR", cv2.WINDOW_NORMAL)
        # cv2.imshow("ROI_BGR", roi_bgr)
        # cv2.waitKey(1)

        # cv2.namedWindow("ROI_HSV", cv2.WINDOW_NORMAL)
        # cv2.imshow("ROI_HSV", roi_hsv)
        # cv2.waitKey(1)

        ##Object-Tracking
        # define range of blue color in HSV
        lower_blue_hsv = np.array([110,50,50])
        upper_blue_hsv = np.array([130,255,255])
        lower_red_hsv = np.array([0,50,50])
        upper_red_hsv = np.array([10,255,255])

        # Threshold the HSV image to get only blue colors
        mask_blue_hsv = cv2.inRange(roi_hsv, lower_blue_hsv, upper_blue_hsv)
        mask_red_hsv = cv2.inRange(roi_hsv, lower_red_hsv, upper_red_hsv)

        thresh_mask_blue_hsv = cv2.threshold(mask_blue_hsv, 200, 255, cv2.THRESH_BINARY)[1]
        thresh_mask_blue_hsv = cv2.erode(thresh_mask_blue_hsv, None, iterations=4)
        thresh_mask_blue_hsv = cv2.dilate(thresh_mask_blue_hsv, None, iterations=8)
        # thresh_mask_red_hsv = cv2.threshold(mask_red_hsv, 200, 255, cv2.THRESH_BINARY)[1]
        # thresh_mask_red_hsv = cv2.erode(thresh_mask_red_hsv, None, iterations=4)
        # thresh_mask_red_hsv = cv2.dilate(thresh_mask_red_hsv, None, iterations=8)
        
        # Bitwise-AND mask and original image
        result_blue_hsv = cv2.bitwise_and(roi_bgr,roi_bgr, mask= thresh_mask_blue_hsv)
        result_red_hsv = cv2.bitwise_and(roi_bgr,roi_bgr, mask= mask_red_hsv)
        #result_bbg_hsv = cv2.bitwise_and(roi_bgr,roi_bgr, mask= fgmask)

        im2,contours_blue,hierarchy = cv2.findContours(thresh_mask_blue_hsv, 1, 2)
        cv2.drawContours(result_blue_hsv, contours_blue, -1, (0,255,0), 3)
        im2,contours_red,hierarchy = cv2.findContours(mask_red_hsv, 1, 2)
        cv2.drawContours(result_red_hsv, contours_red, -1, (0,255,0), 3)

        if len(contours_blue) > 1:
            c_blue = contours_blue[0]
            area_blue = cv2.contourArea(c_blue)
        else: 
            area_blue = 0

        if len(contours_red) > 1:
            c_red = contours_red[0]
            area_red = cv2.contourArea(c_red)
        else: 
            area_red = 0

        # print (area_blue, area_red)

        # mask_hstack_hsv = np.hstack((mask_red_hsv, mask_blue_hsv)) #Stack cv windows horizontally
        # result_hstack_hsv = np.hstack((result_red_hsv, result_blue_hsv)) #Stack cv windows horizontally
        # cv2.namedWindow("MASK_RED_BLUE", cv2.WINDOW_NORMAL)
        # cv2.imshow('MASK_RED_BLUE',mask_hstack_hsv)
        # cv2.namedWindow("RES_RED_BLUE", cv2.WINDOW_NORMAL)
        # cv2.imshow('RES_RED_BLUE',result_hstack_hsv)
        
        # plt.subplot(2,1,1)
        # plt.ylim(0, 2000)
        # plt.autoscale(False, axis='y')
        # plt.plot(i, area_blue, '.-', color = 'blue')
        # plt.subplot(2,1,2)
        # plt.ylim(0, 50)
        # plt.autoscale(False, axis='y')
        # plt.plot(i, area_red, '.-', color = 'red')
        # plt.pause(0.0001)
        
        # cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
        # cv2.imshow("ROI", roi_bgr)
        # cv2.waitKey(1)

        if area_blue>1000 or area_red>=10:
            obj_type = "EV"
        else:
            obj_type = det_class
        #image_pub.publish(bridge.cv2_to_imgmsg(cv_image[y_min:y_max , x_min:x_max], "bgr8"))
        return(obj_type, area_blue, area_red)


def main():
    node = ravev()
    

if __name__ == '__main__':
    main()