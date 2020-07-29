#!/usr/bin/env python

"""
Helper functions and classes will be placed here.
"""

import os

import numpy as np
from darknet_ros_msgs.msg import BoundingBox
from darknet_ros_msgs.msg import BoundingBoxes as BB
from cob_perception_msgs.msg import Detection, DetectionArray, Rect

def create_detection_msg(im, output_dict, category_index, bridge):
    """
    Creates the detection array message

    Args:
    im: (std_msgs_Image) incomming message

    output_dict (dictionary) output of object detection model

    category_index: dictionary of labels (like a lookup table)

    bridge (cv_bridge) : cv bridge object for converting

    Returns:

    msg (cob_perception_msgs/DetectionArray) The message to be sent

    """

    # boxes = output_dict["detection_boxes"]
    # scores = output_dict["detection_scores"]
    # classes = output_dict["detection_classes"]
    # masks = None

    # if 'detection_masks' in output_dict:
    #     masks = output_dict["detection_masks"]

    msg = BoundingBox()

    msg.header = im.header

    scores_above_threshold = np.where(BB.bounding_boxes.probability > 0.5)[0]

    for s in scores_above_threshold:
        # Get the properties

        #bb = BB.bounding_boxes[s,:]
        #sc = BB.bounding_boxes.probability[s]
        #cl = BB.bounding_boxes.Class[s]

        # Create the detection message
        detection = BB()
        detection.header = im.header
        detection.label = BB.bounding_boxes.Class[s]
        detection.id = BB.bounding_boxes.Class[s]
        detection.score = BB.bounding_boxes.probability[s]
        detection.detector = 'Tensorflow object detector'
        detection.mask.roi.x = int((im.width-1) * bb[1])
        detection.mask.roi.y = int((im.height-1) * bb[0])
        detection.mask.roi.width = int((im.width-1) * (bb[3]-bb[1]))
        detection.mask.roi.height = int((im.height-1) * (bb[2]-bb[0]))

        if 'detection_masks' in output_dict:
            detection.mask.mask = \
                bridge.cv2_to_imgmsg(masks[s], "mono8")

            print detection.mask.mask.width


        msg.detections.append(detection)

    return msg
