#!/usr/bin/env python
from __future__ import print_function 
import rospy, roslib
import sys, time
import cv2
import joblib
# import pickle
import cPickle

import message_filters
import numpy as np
from scipy.optimize import linear_sum_assignment
from std_msgs.msg import *
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBoxes
from sort import sort, kalman_tracker
from ravev_function import ravev
from feature_based_classification import classification_models
# import utils
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



class ObjectTrackerNode(object):

    def __init__(self):
            super(ObjectTrackerNode, self).__init__()

            rospy.init_node('object_tracker', anonymous=False)

            self._bridge = CvBridge()
            self.tracker_sort_kalman =  sort.Sort(max_age=1, min_hits=3, use_dlib = False)
            self.tracker_sort_corelation = sort.Sort(max_age=1, min_hits=3, use_dlib = True)


            self.df = pd.DataFrame()
            self.labels = "car"
            #self.box = np.array([0, 0, 1, 1])
            self.cost_threshold = float(10)
            self.ev_count = 0
            self.float_img = np.random.random((4,4))
            self.roi_bgr = np.array(self.float_img * 255, dtype = np.uint8)
            [self.x_min, self.x_max, self.y_min, self.y_max, self.det_class, self.var, self.k, self.det_id, self.var]=[0,0,0,0,0,0,0,0,0]


            # self.read_from_video("traffic.mp4")
            self._ravev = ravev(self.roi_bgr, self.x_min, self.x_max, self.y_min, self.y_max, self.det_class, self.det_id, self.var)
            self.feature_predict = classification_models.FeatureClassifier()
            self.pub_trackers = rospy.Publisher("/output_boxes", BoundingBoxes, queue_size=1)
            # self.ravev_pub_trackers = rospy.Publisher("/ravev_boxes", BoundingBoxes, queue_size=1)
            # self.vid_sub = message_filters.Subscriber("/darknet_ros/detection_image",Image)
            self.vid_sub = message_filters.Subscriber("/videofile/image_raw",Image)
            self.bbox_sub = message_filters.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes)
            self.target_image_size = (150, 150)
            self.nn_model = load_model("ravev_nn_model.h5")
            '''
            self.feature_classifier_dict = {'svm':'weights/feature_based/ravev_svm.pkl',\
                                    'XGBoost':'weights/feature_based/ravev_XGBoost.pkl',\
                                    'decision':'weights/feature_based/ravev_decision_tree.pkl',\
                                    'extra':'weights/feature_based/ravev_extra_tree.pkl',\
                                    'adaboost':'weights/feature_based/ravev_adaboost.pkl',\
                                    'gradient':'weights/feature_based/ravev_gradient_boosting.pkl',\
                                    'random':'weights/feature_based/ravev_random_forrest.pkl',\
                                    'knn':'weights/feature_based/ravev_k_neighbours.pkl'}
            self.feature_model_type = 'adaboost'
            self.feat_model = cPickle.loads('weights/feature_based/ravev_adaboost.pkl')
            '''
            self.tf_graph = tf.get_default_graph()

            ts = message_filters.ApproximateTimeSynchronizer([self.vid_sub, self.bbox_sub],2,0.2,allow_headerless=True)
            ts.registerCallback(self.detection_callback)
            rospy.spin()


    def detection_callback(self, image, bbox):
        timer = cv2.getTickCount()
        cv_image = self._bridge.imgmsg_to_cv2(image, "bgr8")
        
        # Dummy detection
        # TODO: Find a better way
        det_list = np.array([[0, 0, 1, 1, 0.01]])
        ev_count = 0

        if len(bbox.bounding_boxes) > 0:
            # if bbox.bounding_boxes.Class == 'car':
                for i, detection in enumerate(bbox.bounding_boxes):
                    if detection.Class == "car" or "truck" or "bus":
                        # if True:
                            box_x_min = detection.xmin
                            box_y_min = detection.ymin
                            box_x_max = detection.xmax                    
                            box_y_max = detection.ymax
                            width_bbox = box_x_max-box_x_min
                            height_bbox = box_y_max-box_y_min
                            box_score = detection.probability

                            det_list = np.vstack((det_list, \
                                [box_x_min, box_y_min, box_x_max, box_y_max, box_score]))
                    else:
                            del bbox.bounding_boxes[i]

        '''Call the tracker'''
        #tracks = self.tracker_sort_corelation.update(det_list, cv_image)
        tracks = self.tracker_sort_kalman.update(det_list, cv_image)
        #print (tracks)

        '''Copy the detections'''
        detections_copy = bbox.bounding_boxes

        bbox.bounding_boxes = []
        ravev.bounding_boxes = []

        if len(det_list) > 0:

            '''Create cost matrix'''
            C = np.zeros((len(tracks), len(det_list)))
            for i, track in enumerate(tracks):
                for j, det in enumerate(det_list):
                    C[i, j] = np.linalg.norm(det[0:-2] - track[0:-2])


            '''perform linear assignment'''
            row_ind, col_ind = linear_sum_assignment(C)


            for i, j in zip(row_ind, col_ind):
                if j != 0:
                    # if float(C[i, j]) < self.cost_threshold and j != 0:
                        if detections_copy[j-1].Class == 'car' or "truck" or "bus":
                            print("{} -> {} with cost {}".format(tracks[i, 4], detections_copy[j-1].Class, C[i,j]))
                        
                            detections_copy[j-1].id = int(tracks[i, 4])

                            det_x_min=detections_copy[j-1].xmin
                            det_x_max=detections_copy[j-1].xmax
                            det_y_min=detections_copy[j-1].ymin
                            det_y_max=detections_copy[j-1].ymax
                            det_class=detections_copy[j-1].Class
                            
                            roi_bgr=cv_image[det_y_min:det_y_max , det_x_min:det_x_max]
                            
                            det_id=str(int(tracks[i, 4]))
                            var=str(j-1)

                            nn_pred_class = self.nn_predict(roi_bgr)
                            print(nn_pred_class) 
                            
                            '''
                            try:
                                feature_array = self.feature_predict(cv_image, det_x_min, det_x_max, det_y_min, det_y_max, det_class)
                                feature_pred_class = self.feature_based_prediction(feature_array)
                                print(feature_pred_class)
                            except cv2.error:
                                continue
                            '''
                            

                            if float(nn_pred_class[0])>=0.5:
                                print(str('EV'))
                                bbox.bounding_boxes.append(detections_copy[j-1])
                                # cv2.namedWindow("ROI_BGR", cv2.WINDOW_NORMAL)
                                # cv2.imshow("ROI_BGR", roi_bgr)
                                # cv2.waitKey(5)                        
                                
                            # ev_count+=1
                            # print(ev_count)
                            # print(pred_class)

                            # (obj_type, blue_contour, red_contour) = self._ravev.recognize(cv_image, det_x_min, det_x_max, det_y_min, det_y_max, det_class, det_id, var)
                            #print("blue="+str(blue_contour))
                            #print("red="+str(red_contour))


                            # if obj_type == "ev":
                            #     #cv2.namedWindow("ROI_BGR"+k, cv2.WINDOW_NORMAL)
                            #     #cv2.imshow("ROI_BGR"+k, roi_bgr)
                            #     #cv2.waitKey(1)
                            #     ev_count+=1
                            #     print(ev_count) 
                        
                            # bbox.bounding_boxes.append(detections_copy[j-1])
                            

                            # self.df = self.df.append(pd.DataFrame({'i':int(tracks[i, 4]), \
                            #                              'j-1':int(j-1), \
                            #                              'x_min':int(det_x_min), \
                            #                              'x_max':int(det_x_max), \
                            #                              'y_min':int(det_y_min), \
                            #                              'y_max':int(det_y_max), \
                            #                              'obj':str(obj_type), \
                            #                              'blue_area':int(blue_contour), \
                            #                              'red_area':int(red_contour)}, \
                            #                              index=[0]), ignore_index=True, sort=False)
                            
                            # self.df.sort_values(by=['i'])

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            print("FPS:"+str(fps))
            print("------------")
            # self.df.to_excel('excel/output.xlsx')

        else:
            print("No tracked objects!")

        self.pub_trackers.publish(bbox)
        # self.ravev_pub_trackers.publish(ravev)

    def nn_predict(self, img):
        """Run model prediction to classify image as EV and return its probability"""
        
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.target_image_size).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
    
        with self.tf_graph.as_default():
            predictions = self.nn_model.predict(img)
        return predictions

    '''
    def feature_based_prediction(self, feature_array):
        predictions = feat_model.predict(feature_array)
        return predictions
    # def read_from_video(self, video_name):
        
    #     cap = cv2.VideoCapture('video_name')
    #     while(cap.isOpened()):
    #         ret, frame = cap.read()

    #         if frame is not None:
    #             image_message = \
    #                 self._bridge.cv2_to_imgmsg(frame, "bgr8")
    #             self.rgb_callback(image_message)

    #         else:
    #             break

    #     cap.release()
    #     cv2.destroyAllWindows()

    #     #print ("Video has been processed!")

    #     #self.shutdown()
    '''

def main():
    node = ObjectTrackerNode()
    

if __name__ == '__main__':
    main()