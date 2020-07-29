import os,glob
import cv2
import math
from tqdm import tqdm
#import nltk
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm, neighbors
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from xgboost import XGBClassifier 
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import Pipeline



def main():
    #file_list = glob.glob('/*.TXT')
    file_num = 584
    train_df = pd.DataFrame()
    # test_df = pd.DataFrame()

    jobtype = "train" # "train" or "test" or "train_models_only"
    
    # for i in range(0,file_num):
    for i in tqdm(range(584)):
        
        if jobtype == "train":
            train_df = image_processing(jobtype,i,train_df)

            # train_df = train_df[[   'i', 'line', 'img_area', 'Class', 'Class_id', 'x_min', 'x_max', 'y_min', 'y_max', \
            #                         'total_area_hsv_blue', 'total_area_hsv_red', \
            #                         'max_contour_area_hsv_red', 'max_contour_area_hsv_blue', \
            #                         'centroid_X_hsv_red', 'centroid_Y_hsv_red', \
            #                         'centroid_X_hsv_blue', 'centroid_Y_hsv_blue', \
            #                         'total_area_hsv_blue_t20', 'total_area_hsv_red_t20', \
            #                         'max_contour_area_hsv_red_t20', 'max_contour_area_hsv_blue_t20', \
            #                         'centroid_X_hsv_red_t20', 'centroid_Y_hsv_red_t20', \
            #                         'centroid_X_hsv_blue_t20', 'centroid_Y_hsv_blue_t20', \
            #                         'total_area_hsv_thresh_blue', 'total_area_hsv_thresh_red', \
            #                         'max_contour_area_hsv_thresh_red', 'max_contour_area_hsv_thresh_blue', \
            #                         'centroid_X_hsv_thresh_red', 'centroid_Y_hsv_thresh_red', \
            #                         'centroid_X_hsv_thresh_blue', 'centroid_Y_hsv_thresh_blue', \
            #                         'total_area_hsv_thresh_blue_t20', 'total_area_hsv_thresh_red_t20', \
            #                         'max_contour_area_hsv_thresh_red_t20', 'max_contour_area_hsv_thresh_blue_t20', \
            #                         'centroid_X_hsv_thresh_red_t20', 'centroid_Y_hsv_thresh_red_t20', \
            #                         'centroid_X_hsv_thresh_blue_t20', 'centroid_Y_hsv_thresh_blue_t20']]

            train_df = train_df[[   'i', 'line', 'img_area', 'Class', 'Class_id', 'x_min', 'x_max', 'y_min', 'y_max', \
                                    'total_area_hsv_blue', 'total_area_hsv_red', \
                                    'max_contour_area_hsv_red', 'max_contour_area_hsv_blue', \
                                    'centroid_X_hsv_red', 'centroid_Y_hsv_red', \
                                    'centroid_X_hsv_blue', 'centroid_Y_hsv_blue', \
                                    # 'total_area_hsv_blue_t20', 'total_area_hsv_red_t20', \
                                    # 'max_contour_area_hsv_red_t20', 'max_contour_area_hsv_blue_t20', \
                                    # 'centroid_X_hsv_red_t20', 'centroid_Y_hsv_red_t20', \
                                    # 'centroid_X_hsv_blue_t20', 'centroid_Y_hsv_blue_t20', \
                                    'total_area_hsv_thresh_blue', 'total_area_hsv_thresh_red', \
                                    'max_contour_area_hsv_thresh_red', 'max_contour_area_hsv_thresh_blue', \
                                    'centroid_X_hsv_thresh_red', 'centroid_Y_hsv_thresh_red', \
                                    'centroid_X_hsv_thresh_blue', 'centroid_Y_hsv_thresh_blue']]
                                    # 'total_area_hsv_thresh_blue_t20', 'total_area_hsv_thresh_red_t20', \
                                    # 'max_contour_area_hsv_thresh_red_t20', 'max_contour_area_hsv_thresh_blue_t20', \
                                    # 'centroid_X_hsv_thresh_red_t20', 'centroid_Y_hsv_thresh_red_t20', \
                                    # 'centroid_X_hsv_thresh_blue_t20', 'centroid_Y_hsv_thresh_blue_t20']]

            # train_df = train_df[[   'i', 'line', 'img_area', 'Class', 'Class_id', 'x_min', 'x_max', 'y_min', 'y_max', \
            #                         'total_area_hsv_blue', 'total_area_hsv_red', \
            #                         'max_contour_area_hsv_red', 'max_contour_area_hsv_blue', \
            #                         # 'centroid_dist_hsv_red', 'centroid_dist_hsv_blue', \
            #                         'total_area_hsv_blue_t20', 'total_area_hsv_red_t20', \
            #                         'max_contour_area_hsv_red_t20', 'max_contour_area_hsv_blue_t20', \
            #                         # 'centroid_dist_hsv_red_t20', 'centroid_dist_hsv_blue_t20', \
            #                         'total_area_hsv_thresh_blue', 'total_area_hsv_thresh_red', \
            #                         'max_contour_area_hsv_thresh_red', 'max_contour_area_hsv_thresh_blue', \
            #                         # 'centroid_dist_hsv_thresh_red', 'centroid_dist_hsv_thresh_blue', \
            #                         'total_area_hsv_thresh_blue_t20', 'total_area_hsv_thresh_red_t20', \
            #                         'max_contour_area_hsv_thresh_red_t20', 'max_contour_area_hsv_thresh_blue_t20', \
            #                         # 'centroid_dist_hsv_thresh_red_t20', 'centroid_dist_hsv_thresh_blue_t20'
            #                         ]]
            
            train_df.to_excel('/home/abhishek/Documents/projects/RAVEV/ravev/excel/train_output.xlsx')
                       

        elif jobtype == "test":
            test_df = image_processing(jobtype, i, test_df)

    if jobtype == "train" or "train_models_only":
        train_models_df = pd.read_excel('/home/abhishek/Documents/projects/RAVEV/ravev/excel/train_output.xlsx')
        # train_models_df = train_models_df[[   'i', 'line', 'img_area', 'Class', 'Class_id', 'x_min', 'x_max', 'y_min', 'y_max', \
        #             'total_area_hsv_blue', 'total_area_hsv_red', \
        #             'max_contour_area_hsv_red', 'max_contour_area_hsv_blue', \
        #             'centroid_X_hsv_red', 'centroid_Y_hsv_red', \
        #             'centroid_X_hsv_blue', 'centroid_Y_hsv_blue', \
        #             'total_area_hsv_blue_t20', 'total_area_hsv_red_t20', \
        #             'max_contour_area_hsv_red_t20', 'max_contour_area_hsv_blue_t20', \
        #             'centroid_X_hsv_red_t20', 'centroid_Y_hsv_red_t20', \
        #             'centroid_X_hsv_blue_t20', 'centroid_Y_hsv_blue_t20', \
        #             'total_area_hsv_thresh_blue', 'total_area_hsv_thresh_red', \
        #             'max_contour_area_hsv_thresh_red', 'max_contour_area_hsv_thresh_blue', \
        #             'centroid_X_hsv_thresh_red', 'centroid_Y_hsv_thresh_red', \
        #             'centroid_X_hsv_thresh_blue', 'centroid_Y_hsv_thresh_blue', \
        #             'total_area_hsv_thresh_blue_t20', 'total_area_hsv_thresh_red_t20', \
        #             'max_contour_area_hsv_thresh_red_t20', 'max_contour_area_hsv_thresh_blue_t20', \
        #             'centroid_X_hsv_thresh_red_t20', 'centroid_Y_hsv_thresh_red_t20', \
        #             'centroid_X_hsv_thresh_blue_t20', 'centroid_Y_hsv_thresh_blue_t20']]

        train_models_df = train_models_df[[   'i', 'line', 'img_area', 'Class', 'Class_id', 'x_min', 'x_max', 'y_min', 'y_max', \
                    'total_area_hsv_blue', 'total_area_hsv_red', \
                    'max_contour_area_hsv_red', 'max_contour_area_hsv_blue', \
                    'centroid_X_hsv_red', 'centroid_Y_hsv_red', \
                    'centroid_X_hsv_blue', 'centroid_Y_hsv_blue', \
                    # 'total_area_hsv_blue_t20', 'total_area_hsv_red_t20', \
                    # 'max_contour_area_hsv_red_t20', 'max_contour_area_hsv_blue_t20', \
                    # 'centroid_X_hsv_red_t20', 'centroid_Y_hsv_red_t20', \
                    # 'centroid_X_hsv_blue_t20', 'centroid_Y_hsv_blue_t20', \
                    'total_area_hsv_thresh_blue', 'total_area_hsv_thresh_red', \
                    'max_contour_area_hsv_thresh_red', 'max_contour_area_hsv_thresh_blue', \
                    'centroid_X_hsv_thresh_red', 'centroid_Y_hsv_thresh_red', \
                    'centroid_X_hsv_thresh_blue', 'centroid_Y_hsv_thresh_blue']]
                    # 'total_area_hsv_thresh_blue_t20', 'total_area_hsv_thresh_red_t20', \
                    # 'max_contour_area_hsv_thresh_red_t20', 'max_contour_area_hsv_thresh_blue_t20', \
                    # 'centroid_X_hsv_thresh_red_t20', 'centroid_Y_hsv_thresh_red_t20', \
                    # 'centroid_X_hsv_thresh_blue_t20', 'centroid_Y_hsv_thresh_blue_t20']]

        # train_models_df = train_models_df[[   'i', 'line', 'img_area', 'Class', 'Class_id', 'x_min', 'x_max', 'y_min', 'y_max', \
        #             'total_area_hsv_blue', 'total_area_hsv_red', \
        #             'max_contour_area_hsv_red', 'max_contour_area_hsv_blue', \
        #             # 'centroid_dist_hsv_red', 'centroid_dist_hsv_blue', \
        #             'total_area_hsv_blue_t20', 'total_area_hsv_red_t20', \
        #             'max_contour_area_hsv_red_t20', 'max_contour_area_hsv_blue_t20', \
        #             # 'centroid_dist_hsv_red_t20', 'centroid_dist_hsv_blue_t20', \
        #             'total_area_hsv_thresh_blue', 'total_area_hsv_thresh_red', \
        #             'max_contour_area_hsv_thresh_red', 'max_contour_area_hsv_thresh_blue', \
        #             # 'centroid_dist_hsv_thresh_red', 'centroid_dist_hsv_thresh_blue', \
        #             'total_area_hsv_thresh_blue_t20', 'total_area_hsv_thresh_red_t20', \
        #             'max_contour_area_hsv_thresh_red_t20', 'max_contour_area_hsv_thresh_blue_t20', \
        #             # 'centroid_dist_hsv_thresh_red_t20', 'centroid_dist_hsv_thresh_blue_t20']]
        #             ]]
        
        X_train = train_models_df.drop(columns=['i', 'line', 'x_min', 'x_max',\
                                        'y_min', 'y_max', 'Class', 'Class_id', 'img_area'])
        Y_train = train_models_df['Class_id']

    
    train_svm(X_train, Y_train)
    train_decision_tree(X_train, Y_train)
    train_extra_tree(X_train, Y_train)
    train_adaboost(X_train, Y_train)
    train_gradient_boosting(X_train, Y_train)
    train_random_forrest(X_train, Y_train)
    train_kNeighbours(X_train, Y_train)
    train_XGBoost(X_train, Y_train)
    



def image_processing(jobtype, i, df):
    
    imgfile = cv2.imread('/home/abhishek/Documents/projects/RAVEV/database/test'+str(i)+'.jpg')
    txtfile = open('/home/abhishek/Documents/projects/RAVEV/database/test'+str(i)+'.txt','r+')

    [height, width, channels] = imgfile.shape
    imgfile_area = height*width

    line_num=0
    for line in txtfile.readlines():
        line_num += 1
        [Class_id, x, y, w, h] = line.split(' ')

        if Class_id == '1':
            Class = 'EV'
        elif Class_id =='2':
            Class = 'car'

        #print Class
        x_min = int( (float(width)*float(x)) - ((float(width)*float(w))/2) )
        y_min = int( float(height)*float(y) - ((float(height)*float(h))/2) )
        x_max = x_min + int(float(width)*float(w))
        y_max = y_min + int(float(height)*float(h))
        x_mid = x_max/2
        y_max_20 = y_min + 0.3*(y_max - y_min)
        #print x_min, x_max, y_min, y_max
        roi_bgr = imgfile[int(y_min):int(y_max) , int(x_min):int(x_max)]
        roi_bgr_t20 = imgfile[int(y_min):int(y_max_20) , int(x_min):int(x_max)]
        #print line_num

        # # Convert BGR to HSV
        # roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        # roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        try:
            ##**** Explain Method 1 & 2 ****##
            total_area_hsv_red, max_contour_area_hsv_red, centroid_X_hsv_red, centroid_Y_hsv_red = find_contour_area('red',roi_bgr,'1')
            total_area_hsv_blue, max_contour_area_hsv_blue, centroid_X_hsv_blue, centroid_Y_hsv_blue = find_contour_area('blue',roi_bgr,'1') 
            centroid_dist_hsv_red = math.sqrt(int(centroid_X_hsv_red)^2 + int(centroid_Y_hsv_red)^2)
            centroid_dist_hsv_blue = math.sqrt(int(centroid_X_hsv_blue)^2 + int(centroid_Y_hsv_blue)^2)

            total_area_hsv_red_t20, max_contour_area_hsv_red_t20, centroid_X_hsv_red_t20, centroid_Y_hsv_red_t20 = find_contour_area('red',roi_bgr_t20,'1')
            total_area_hsv_blue_t20, max_contour_area_hsv_blue_t20, centroid_X_hsv_blue_t20, centroid_Y_hsv_blue_t20 = find_contour_area('blue',roi_bgr_t20,'1')
            centroid_dist_hsv_red_t20 = math.sqrt(int(centroid_X_hsv_red_t20)^2 + int(centroid_Y_hsv_red_t20)^2)
            centroid_dist_hsv_blue_t20 = math.sqrt(int(centroid_X_hsv_blue_t20)^2 + int(centroid_Y_hsv_blue_t20)^2)


            'binary thresholding'
            total_area_hsv_thresh_red, max_contour_area_hsv_thresh_red, centroid_X_hsv_thresh_red, centroid_Y_hsv_thresh_red = find_contour_area('red',roi_bgr,'2')
            total_area_hsv_thresh_blue, max_contour_area_hsv_thresh_blue, centroid_X_hsv_thresh_blue, centroid_Y_hsv_thresh_blue = find_contour_area('blue',roi_bgr,'2')
            centroid_dist_hsv_thresh_red = math.sqrt(int(centroid_X_hsv_thresh_red)^2 + int(centroid_Y_hsv_thresh_red)^2)
            centroid_dist_hsv_thresh_blue = math.sqrt(int(centroid_X_hsv_thresh_blue)^2 + int(centroid_Y_hsv_thresh_blue)^2) 

            total_area_hsv_thresh_red_t20, max_contour_area_hsv_thresh_red_t20, centroid_X_hsv_thresh_red_t20, centroid_Y_hsv_thresh_red_t20 = find_contour_area('red',roi_bgr_t20,'2')
            total_area_hsv_thresh_blue_t20, max_contour_area_hsv_thresh_blue_t20, centroid_X_hsv_thresh_blue_t20, centroid_Y_hsv_thresh_blue_t20 = find_contour_area('blue',roi_bgr_t20,'2')
            centroid_dist_hsv_thresh_red_t20 = math.sqrt(int(centroid_X_hsv_thresh_red_t20)^2 + int(centroid_Y_hsv_thresh_red_t20)^2)
            centroid_dist_hsv_thresh_blue_t20 = math.sqrt(int(centroid_X_hsv_thresh_blue_t20)^2 + int(centroid_Y_hsv_thresh_blue_t20)^2) 


        except cv2.error:
            continue

        if jobtype=="train":
            # df = df.append(pd.DataFrame({   'i':int(i), 'line':int(line_num), 'img_area':int(imgfile_area), 'Class':str(Class), 'Class_id':str(Class_id),\
            #                                 'x_min':int(x_min), 'x_max':int(x_max), 'y_min':int(y_min), 'y_max':int(y_max), \
            #                                 'total_area_hsv_blue':float(total_area_hsv_blue), 'total_area_hsv_red':float(total_area_hsv_red), \
            #                                 'max_contour_area_hsv_red':float(max_contour_area_hsv_red), 'max_contour_area_hsv_blue':float(max_contour_area_hsv_blue), \
            #                                 'centroid_X_hsv_red':float(centroid_X_hsv_red), 'centroid_Y_hsv_red':float(centroid_Y_hsv_red), \
            #                                 'centroid_X_hsv_blue':float(centroid_X_hsv_blue), 'centroid_Y_hsv_blue':float(centroid_Y_hsv_blue), \
            #                                 'total_area_hsv_blue_t20':float(total_area_hsv_blue_t20), 'total_area_hsv_red_t20':float(total_area_hsv_red_t20), \
            #                                 'max_contour_area_hsv_red_t20':float(max_contour_area_hsv_red_t20), 'max_contour_area_hsv_blue_t20':float(max_contour_area_hsv_blue_t20), \
            #                                 'centroid_X_hsv_red_t20':float(centroid_X_hsv_red_t20), 'centroid_Y_hsv_red_t20':float(centroid_Y_hsv_red_t20), \
            #                                 'centroid_X_hsv_blue_t20':float(centroid_X_hsv_blue_t20), 'centroid_Y_hsv_blue_t20':float(centroid_Y_hsv_blue_t20), \
            #                                 'total_area_hsv_thresh_blue':float(total_area_hsv_thresh_blue), 'total_area_hsv_thresh_red':float(total_area_hsv_thresh_red), \
            #                                 'max_contour_area_hsv_thresh_red':float(max_contour_area_hsv_thresh_red), 'max_contour_area_hsv_thresh_blue':float(max_contour_area_hsv_thresh_blue), \
            #                                 'centroid_X_hsv_thresh_red':float(centroid_X_hsv_thresh_red), 'centroid_Y_hsv_thresh_red':float(centroid_Y_hsv_thresh_red), \
            #                                 'centroid_X_hsv_thresh_blue':float(centroid_X_hsv_thresh_blue), 'centroid_Y_hsv_thresh_blue':float(centroid_Y_hsv_thresh_blue), \
            #                                 'total_area_hsv_thresh_blue_t20':float(total_area_hsv_thresh_blue_t20), 'total_area_hsv_thresh_red_t20':float(total_area_hsv_thresh_red_t20), \
            #                                 'max_contour_area_hsv_thresh_red_t20':float(max_contour_area_hsv_thresh_red_t20), 'max_contour_area_hsv_thresh_blue_t20':float(max_contour_area_hsv_thresh_blue_t20), \
            #                                 'centroid_X_hsv_thresh_red_t20':float(centroid_X_hsv_thresh_red_t20), 'centroid_Y_hsv_thresh_red_t20':float(centroid_Y_hsv_thresh_red_t20), \
            #                                 'centroid_X_hsv_thresh_blue_t20':float(centroid_X_hsv_thresh_blue_t20), 'centroid_Y_hsv_thresh_blue_t20':float(centroid_Y_hsv_thresh_blue_t20)}, \
            #                                 index=[0]), ignore_index=True, sort=False)
            
            df = df.append(pd.DataFrame({   'i':int(i), 'line':int(line_num), 'img_area':int(imgfile_area), 'Class':str(Class), 'Class_id':str(Class_id),\
                                            'x_min':int(x_min), 'x_max':int(x_max), 'y_min':int(y_min), 'y_max':int(y_max), \
                                            'total_area_hsv_blue':float(total_area_hsv_blue), 'total_area_hsv_red':float(total_area_hsv_red), \
                                            'max_contour_area_hsv_red':float(max_contour_area_hsv_red), 'max_contour_area_hsv_blue':float(max_contour_area_hsv_blue), \
                                            'centroid_X_hsv_red':float(centroid_X_hsv_red), 'centroid_Y_hsv_red':float(centroid_Y_hsv_red), \
                                            'centroid_X_hsv_blue':float(centroid_X_hsv_blue), 'centroid_Y_hsv_blue':float(centroid_Y_hsv_blue), \
                                            # 'total_area_hsv_blue_t20':float(total_area_hsv_blue_t20), 'total_area_hsv_red_t20':float(total_area_hsv_red_t20), \
                                            # 'max_contour_area_hsv_red_t20':float(max_contour_area_hsv_red_t20), 'max_contour_area_hsv_blue_t20':float(max_contour_area_hsv_blue_t20), \
                                            # 'centroid_X_hsv_red_t20':float(centroid_X_hsv_red_t20), 'centroid_Y_hsv_red_t20':float(centroid_Y_hsv_red_t20), \
                                            # 'centroid_X_hsv_blue_t20':float(centroid_X_hsv_blue_t20), 'centroid_Y_hsv_blue_t20':float(centroid_Y_hsv_blue_t20), \
                                            'total_area_hsv_thresh_blue':float(total_area_hsv_thresh_blue), 'total_area_hsv_thresh_red':float(total_area_hsv_thresh_red), \
                                            'max_contour_area_hsv_thresh_red':float(max_contour_area_hsv_thresh_red), 'max_contour_area_hsv_thresh_blue':float(max_contour_area_hsv_thresh_blue), \
                                            'centroid_X_hsv_thresh_red':float(centroid_X_hsv_thresh_red), 'centroid_Y_hsv_thresh_red':float(centroid_Y_hsv_thresh_red), \
                                            'centroid_X_hsv_thresh_blue':float(centroid_X_hsv_thresh_blue), 'centroid_Y_hsv_thresh_blue':float(centroid_Y_hsv_thresh_blue)}, \
                                            # 'total_area_hsv_thresh_blue_t20':float(total_area_hsv_thresh_blue_t20), 'total_area_hsv_thresh_red_t20':float(total_area_hsv_thresh_red_t20), \
                                            # 'max_contour_area_hsv_thresh_red_t20':float(max_contour_area_hsv_thresh_red_t20), 'max_contour_area_hsv_thresh_blue_t20':float(max_contour_area_hsv_thresh_blue_t20), \
                                            # 'centroid_X_hsv_thresh_red_t20':float(centroid_X_hsv_thresh_red_t20), 'centroid_Y_hsv_thresh_red_t20':float(centroid_Y_hsv_thresh_red_t20), \
                                            # 'centroid_X_hsv_thresh_blue_t20':float(centroid_X_hsv_thresh_blue_t20), 'centroid_Y_hsv_thresh_blue_t20':float(centroid_Y_hsv_thresh_blue_t20)}, \
                                            index=[0]), ignore_index=True, sort=False)
        
        #     df = df.append(pd.DataFrame({   'i':int(i), 'line':int(line_num), 'img_area':int(imgfile_area), 'Class':str(Class), 'Class_id':str(Class_id),\
        #                                     'x_min':int(x_min), 'x_max':int(x_max), 'y_min':int(y_min), 'y_max':int(y_max), \
        #                                     'total_area_hsv_blue':float(total_area_hsv_blue), 'total_area_hsv_red':float(total_area_hsv_red), \
        #                                     'max_contour_area_hsv_red':float(max_contour_area_hsv_red), 'max_contour_area_hsv_blue':float(max_contour_area_hsv_blue), \
        #                                     # 'centroid_dist_hsv_red':float(centroid_dist_hsv_red), 'centroid_dist_hsv_blue':float(centroid_dist_hsv_blue), \
        #                                     'total_area_hsv_blue_t20':float(total_area_hsv_blue_t20), 'total_area_hsv_red_t20':float(total_area_hsv_red_t20), \
        #                                     'max_contour_area_hsv_red_t20':float(max_contour_area_hsv_red_t20), 'max_contour_area_hsv_blue_t20':float(max_contour_area_hsv_blue_t20), \
        #                                     # 'centroid_dist_hsv_red_t20':float(centroid_dist_hsv_red_t20), 'centroid_dist_hsv_blue_t20':float(centroid_dist_hsv_blue_t20), \
        #                                     'total_area_hsv_thresh_blue':float(total_area_hsv_thresh_blue), 'total_area_hsv_thresh_red':float(total_area_hsv_thresh_red), \
        #                                     'max_contour_area_hsv_thresh_red':float(max_contour_area_hsv_thresh_red), 'max_contour_area_hsv_thresh_blue':float(max_contour_area_hsv_thresh_blue), \
        #                                     # 'centroid_dist_hsv_thresh_red':float(centroid_dist_hsv_thresh_red), 'centroid_dist_hsv_thresh_blue':float(centroid_dist_hsv_thresh_blue), \
        #                                     'total_area_hsv_thresh_blue_t20':float(total_area_hsv_thresh_blue_t20), 'total_area_hsv_thresh_red_t20':float(total_area_hsv_thresh_red_t20), \
        #                                     'max_contour_area_hsv_thresh_red_t20':float(max_contour_area_hsv_thresh_red_t20), 'max_contour_area_hsv_thresh_blue_t20':float(max_contour_area_hsv_thresh_blue_t20)}, \
        #                                     # 'centroid_dist_hsv_thresh_red_t20':float(centroid_dist_hsv_thresh_red_t20), 'centroid_dist_hsv_thresh_blue_t20':float(centroid_dist_hsv_thresh_blue_t20)}, \
        #                                     index=[0]), ignore_index=True, sort=False)

        elif jobtype=="test":
            df = pd.DataFrame({             'total_area_hsv_blue':float(total_area_hsv_blue), 'total_area_hsv_red':float(total_area_hsv_red), \
                                            'max_contour_area_hsv_red':float(max_contour_area_hsv_red), 'max_contour_area_hsv_blue':float(max_contour_area_hsv_blue), \
                                            'centroid_X_hsv_red':float(centroid_X_hsv_red), 'centroid_Y_hsv_red':float(centroid_Y_hsv_red), \
                                            'centroid_X_hsv_blue':float(centroid_X_hsv_blue), 'centroid_Y_hsv_blue':float(centroid_Y_hsv_blue), \
                                            'total_area_hsv_blue_t20':float(total_area_hsv_blue_t20), 'total_area_hsv_red_t20':float(total_area_hsv_red_t20), \
                                            'max_contour_area_hsv_red_t20':float(max_contour_area_hsv_red_t20), 'max_contour_area_hsv_blue_t20':float(max_contour_area_hsv_blue_t20), \
                                            'centroid_X_hsv_red_t20':float(centroid_X_hsv_red_t20), 'centroid_Y_hsv_red_t20':float(centroid_Y_hsv_red_t20), \
                                            'centroid_X_hsv_blue_t20':float(centroid_X_hsv_blue_t20), 'centroid_Y_hsv_blue_t20':float(centroid_Y_hsv_blue_t20), \
                                            'total_area_hsv_thresh_blue':float(total_area_hsv_thresh_blue), 'total_area_hsv_thresh_red':float(total_area_hsv_thresh_red), \
                                            'max_contour_area_hsv_thresh_red':float(max_contour_area_hsv_thresh_red), 'max_contour_area_hsv_thresh_blue':float(max_contour_area_hsv_thresh_blue), \
                                            'centroid_X_hsv_thresh_red':float(centroid_X_hsv_thresh_red), 'centroid_Y_hsv_thresh_red':float(centroid_Y_hsv_thresh_red), \
                                            'centroid_X_hsv_thresh_blue':float(centroid_X_hsv_thresh_blue), 'centroid_Y_hsv_thresh_blue':float(centroid_Y_hsv_thresh_blue), \
                                            'total_area_hsv_thresh_blue_t20':float(total_area_hsv_thresh_blue_t20), 'total_area_hsv_thresh_red_t20':float(total_area_hsv_thresh_red_t20), \
                                            'max_contour_area_hsv_thresh_red_t20':float(max_contour_area_hsv_thresh_red_t20), 'max_contour_area_hsv_thresh_blue_t20':float(max_contour_area_hsv_thresh_blue_t20), \
                                            'centroid_X_hsv_thresh_red_t20':float(centroid_X_hsv_thresh_red_t20), 'centroid_Y_hsv_thresh_red_t20':float(centroid_Y_hsv_thresh_red_t20), \
                                            'centroid_X_hsv_thresh_blue_t20':float(centroid_X_hsv_thresh_blue_t20), 'centroid_Y_hsv_thresh_blue_t20':float(centroid_Y_hsv_thresh_blue_t20)}, \
                                            index=[0])
            
            df = pd.DataFrame({             'total_area_hsv_blue':float(total_area_hsv_blue), 'total_area_hsv_red':float(total_area_hsv_red), \
                                            'max_contour_area_hsv_red':float(max_contour_area_hsv_red), 'max_contour_area_hsv_blue':float(max_contour_area_hsv_blue), \
                                            'centroid_X_hsv_red':float(centroid_X_hsv_red), 'centroid_Y_hsv_red':float(centroid_Y_hsv_red), \
                                            'centroid_X_hsv_blue':float(centroid_X_hsv_blue), 'centroid_Y_hsv_blue':float(centroid_Y_hsv_blue), \
                                            # 'total_area_hsv_blue_t20':float(total_area_hsv_blue_t20), 'total_area_hsv_red_t20':float(total_area_hsv_red_t20), \
                                            # 'max_contour_area_hsv_red_t20':float(max_contour_area_hsv_red_t20), 'max_contour_area_hsv_blue_t20':float(max_contour_area_hsv_blue_t20), \
                                            # 'centroid_X_hsv_red_t20':float(centroid_X_hsv_red_t20), 'centroid_Y_hsv_red_t20':float(centroid_Y_hsv_red_t20), \
                                            # 'centroid_X_hsv_blue_t20':float(centroid_X_hsv_blue_t20), 'centroid_Y_hsv_blue_t20':float(centroid_Y_hsv_blue_t20), \
                                            'total_area_hsv_thresh_blue':float(total_area_hsv_thresh_blue), 'total_area_hsv_thresh_red':float(total_area_hsv_thresh_red), \
                                            'max_contour_area_hsv_thresh_red':float(max_contour_area_hsv_thresh_red), 'max_contour_area_hsv_thresh_blue':float(max_contour_area_hsv_thresh_blue), \
                                            'centroid_X_hsv_thresh_red':float(centroid_X_hsv_thresh_red), 'centroid_Y_hsv_thresh_red':float(centroid_Y_hsv_thresh_red), \
                                            'centroid_X_hsv_thresh_blue':float(centroid_X_hsv_thresh_blue), 'centroid_Y_hsv_thresh_blue':float(centroid_Y_hsv_thresh_blue)}, \
                                            # 'total_area_hsv_thresh_blue_t20':float(total_area_hsv_thresh_blue_t20), 'total_area_hsv_thresh_red_t20':float(total_area_hsv_thresh_red_t20), \
                                            # 'max_contour_area_hsv_thresh_red_t20':float(max_contour_area_hsv_thresh_red_t20), 'max_contour_area_hsv_thresh_blue_t20':float(max_contour_area_hsv_thresh_blue_t20), \
                                            # 'centroid_X_hsv_thresh_red_t20':float(centroid_X_hsv_thresh_red_t20), 'centroid_Y_hsv_thresh_red_t20':float(centroid_Y_hsv_thresh_red_t20), \
                                            # 'centroid_X_hsv_thresh_blue_t20':float(centroid_X_hsv_thresh_blue_t20), 'centroid_Y_hsv_thresh_blue_t20':float(centroid_Y_hsv_thresh_blue_t20)}, \
                                            index=[0])
        
            # df = pd.DataFrame({             'total_area_hsv_blue':float(total_area_hsv_blue), 'total_area_hsv_red':float(total_area_hsv_red), \
            #                                 'max_contour_area_hsv_red':float(max_contour_area_hsv_red), 'max_contour_area_hsv_blue':float(max_contour_area_hsv_blue), \
            #                                 # 'centroid_dist_hsv_red':float(centroid_dist_hsv_red), 'centroid_dist_hsv_blue':float(centroid_dist_hsv_blue), \
            #                                 'total_area_hsv_blue_t20':float(total_area_hsv_blue_t20), 'total_area_hsv_red_t20':float(total_area_hsv_red_t20), \
            #                                 'max_contour_area_hsv_red_t20':float(max_contour_area_hsv_red_t20), 'max_contour_area_hsv_blue_t20':float(max_contour_area_hsv_blue_t20), \
            #                                 # 'centroid_dist_hsv_red_t20':float(centroid_dist_hsv_red_t20), 'centroid_dist_hsv_blue_t20':float(centroid_dist_hsv_blue_t20), \
            #                                 'total_area_hsv_thresh_blue':float(total_area_hsv_thresh_blue), 'total_area_hsv_thresh_red':float(total_area_hsv_thresh_red), \
            #                                 'max_contour_area_hsv_thresh_red':float(max_contour_area_hsv_thresh_red), 'max_contour_area_hsv_thresh_blue':float(max_contour_area_hsv_thresh_blue), \
            #                                 # 'centroid_dist_hsv_thresh_red':float(centroid_dist_hsv_thresh_red), 'centroid_dist_hsv_thresh_blue':float(centroid_dist_hsv_thresh_blue), \
            #                                 'total_area_hsv_thresh_blue_t20':float(total_area_hsv_thresh_blue_t20), 'total_area_hsv_thresh_red_t20':float(total_area_hsv_thresh_red_t20), \
            #                                 'max_contour_area_hsv_thresh_red_t20':float(max_contour_area_hsv_thresh_red_t20), 'max_contour_area_hsv_thresh_blue_t20':float(max_contour_area_hsv_thresh_blue_t20)}, \
            #                                 # 'centroid_dist_hsv_thresh_red_t20':float(centroid_dist_hsv_thresh_red_t20), 'centroid_dist_hsv_thresh_blue_t20':float(centroid_dist_hsv_thresh_blue_t20)}, \
            #                                 index=[0])

            # X_test = [[ float(total_area_hsv_blue), float(total_area_hsv_red), \
            #             float(total_area_hsv_blue_t20), float(total_area_hsv_red_t20), \
            #             float(total_area_hsv_thresh_blue), float(total_area_hsv_thresh_red), \
            #             float(total_area_hsv_thresh_blue_t20), float(total_area_hsv_thresh_red_t20) ]]
            # classify_svm(X_test)
            

        # while(True):
        #     cv2.namedWindow("ROI_BGR", cv2.WINDOW_NORMAL)
        #     cv2.imshow("ROI_BGR", roi_bgr)
        #     cv2.namedWindow("MASK_RED_BLUE", cv2.WINDOW_NORMAL)
        #     cv2.imshow('MASK_RED_BLUE',mask_hstack_hsv)
        #     cv2.namedWindow("RES_RED_BLUE", cv2.WINDOW_NORMAL)
        #     cv2.imshow('RES_RED_BLUE',result_hstack_hsv)
        #     cv2.namedWindow("THRESH_BINARY", cv2.WINDOW_NORMAL)
        #     cv2.imshow('THRESH_BINARY',roi_thresh_binary)
        #     cv2.namedWindow("MASK_THRESH_RED_BLUE", cv2.WINDOW_NORMAL)
        #     cv2.imshow('MASK_THRESH_RED_BLUE',mask_hstack_thresh_hsv)
        #     cv2.namedWindow("RES_THRESH_RED_BLUE", cv2.WINDOW_NORMAL)
        #     cv2.imshow('RES_THRESH_RED_BLUE',result_hstack_thresh_hsv)
        #     # cv2.namedWindow("Adaptive Mean Thresholding", cv2.WINDOW_NORMAL)
        #     # cv2.imshow('Adaptive Mean Thresholding',thresh2)
        #     # cv2.namedWindow("Adaptive Gaussian Thresholding", cv2.WINDOW_NORMAL)
        #     # cv2.imshow('Adaptive Gaussian Thresholding',thresh3)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

    txtfile.close()
    # print i
    return df



def find_contour_area(color, image, method):
       
    if color=='red':   
        lower_bound = np.array([0,50,50])
        upper_bound = np.array([10,255,255])
    elif color=='blue':
        lower_bound = np.array([110,50,50])
        upper_bound = np.array([130,255,255])


    if method == '1':
        roi_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(roi_hsv, lower_bound, upper_bound)

        thresh_mask_hsv = cv2.threshold(mask_hsv, 127, 255, cv2.THRESH_BINARY)[1]
        # thresh_mask_hsv = cv2.erode(thresh_mask_hsv, None, iterations=4)
        # thresh_mask_hsv = cv2.dilate(thresh_mask_hsv, None, iterations=8)
        masked_image = cv2.bitwise_and(image, image, mask= thresh_mask_hsv) ##**** if required to display ****##
        im2,contours,hierarchy = cv2.findContours(thresh_mask_hsv, 1, 2)
        cv2.drawContours(masked_image, contours, -1, (0,255,0), 0)

    elif method == '2':
        image_blur = cv2.medianBlur(image,5)
        thresh_image_blur = cv2.threshold(image_blur, 127, 255, cv2.THRESH_BINARY)[1]
        roi_hsv = cv2.cvtColor(thresh_image_blur, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(roi_hsv, lower_bound, upper_bound)
        masked_image = cv2.bitwise_and(thresh_image_blur, thresh_image_blur, mask= mask_hsv) ##**** if required to display ****##
        im2,contours,hierarchy = cv2.findContours(mask_hsv, 1, 2)
        cv2.drawContours(masked_image, contours, -1, (0,255,0), 0)
     
    
    total_area = 0
    if len(contours) > 0:
        M = cv2.moments(contours[0])
        max_contour_moment = cv2.moments(contours[0])
        max_contour_area = cv2.contourArea(contours[0])
        for k in range(len(contours)):
            total_area += cv2.contourArea(contours[k])
            if cv2.contourArea(contours[k]) >= max_contour_area:
                max_contour_area = cv2.contourArea(contours[k])
            # if largest_contour_moment <= cv2.moments(contours[k]):
                max_contour_moment = cv2.moments(contours[k])

            if max_contour_moment['m00']>0:
                centroid_X = float(max_contour_moment['m10']/max_contour_moment['m00'])
                centroid_Y = float(max_contour_moment['m01']/max_contour_moment['m00'])
            else:
                centroid_X, centroid_Y = [0,0]
    else:
        max_contour_moment, max_contour_area, centroid_X, centroid_Y = [0,0,0,0]

    return total_area, max_contour_area, centroid_X, centroid_Y

def classify_svm(X_test):
    clf = joblib.load('/home/abhishek/Documents/projects/RAVEV/ravev/weights/feature_based/ravev_svm.pkl')
    Y_pred = clf.predict(X_test)
    print Y_pred
    return Y_pred

def train_svm(X, Y):
    # split data into train and test sets
    seed = 7
    test_size = 0.25
    target_names = ['ev','car']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    # fit model no training data
    model = svm.SVC(kernel='poly', max_iter=50000)  
    # model = SelectFromModel(model)
    # model = RFE(model, 5, step=1)
    model.fit(X_train, Y_train)

    joblib.dump(model, '/home/abhishek/Documents/projects/RAVEV/ravev/weights/feature_based/ravev_svm.pkl')
    scores = cross_val_score(model, X_train, Y_train)
    Cross_val_score = scores.mean()

    # make predictions for test data
    Y_pred = model.predict(X_test)
    predictions = [round(value) for value in Y_pred]

    
    # X_train_new = model.transform(X_train)

    # evaluate predictions
    accuracy = accuracy_score(Y_test, predictions)
    Precision_score = precision_score(Y_test, predictions)
    Recall_score = recall_score(Y_test, predictions)
    print("Accuracy SVM : %.2f%%" % (accuracy * 100.0))
    print("Cross Validation Score SVM: %.2f%%" % (Cross_val_score * 100.0))
    print("Precision Score SVM: %.2f%%" % (Precision_score * 100.0))
    print("Recall Score SVM: %.2f%%" % (Recall_score * 100.0))
    # print(classification_report(Y_test, Y_pred, target_names=target_names))
    # print(confusion_matrix(Y_test, Y_pred))
    # print(X_train.shape)
    # print(model.shape)
    # print(model.feature_importances_)
    # plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    # plt.show()

def train_XGBoost(X,Y):
    # split data into train and test sets
    seed = 7
    test_size = 0.25
    target_names = ['ev','car']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model no training data
    model = XGBClassifier()
    # model = SelectFromModel(model)
    # model = RFE(model, 5, step=1)
    model.fit(X_train, Y_train)
    joblib.dump(model, '/home/abhishek/Documents/projects/RAVEV/ravev/weights/feature_based/ravev_XGBoost.pkl')
    scores = cross_val_score(model, X_train, Y_train)
    Cross_val_score = scores.mean()

    # make predictions for test data
    Y_pred = model.predict(X_test)
    predictions = [round(value) for value in Y_pred]

    # model = SelectFromModel(model)
    # X_train_new = model.transform(X_train)

    # evaluate predictions
    accuracy = accuracy_score(Y_test, predictions)
    Precision_score = precision_score(Y_test, predictions)
    Recall_score = recall_score(Y_test, predictions)
    print("Accuracy XGBoost : %.2f%%" % (accuracy * 100.0))
    print("Cross Validation Score XGBoost: %.2f%%" % (Cross_val_score * 100.0))
    print("Precision Score XGBoost: %.2f%%" % (Precision_score * 100.0))
    print("Recall Score XGBoost: %.2f%%" % (Recall_score * 100.0))
    print(classification_report(Y_test, Y_pred, target_names=target_names))
    print(confusion_matrix(Y_test, Y_pred))
    # print(X_train.shape)
    # print(X_train_new.shape)
    print(model.feature_importances_)
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.show()

def train_decision_tree(X,Y):
    # split data into train and test sets
    seed = 7
    test_size = 0.25
    target_names = ['ev','car']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    # fit model no training data
    #X, y = make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)
    model = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    # model = SelectFromModel(model)
    # model = RFE(model, 5, step=1)
    model.fit(X_train, Y_train)
    joblib.dump(model, '/home/abhishek/Documents/projects/RAVEV/ravev/weights/feature_based/ravev_decision_tree.pkl')
    scores = cross_val_score(model, X_train, Y_train)
    Cross_val_score = scores.mean()
  
    # make predictions for test data
    Y_pred = model.predict(X_test)
    predictions = [round(value) for value in Y_pred]

    # model = SelectFromModel(model, prefit=True)
    # X_train_new = model.transform(X_train)

    # evaluate predictions
    accuracy = accuracy_score(Y_test, predictions)
    Precision_score = precision_score(Y_test, predictions)
    Recall_score = recall_score(Y_test, predictions)
    print("Accuracy Decision Tree: %.2f%%" % (accuracy * 100.0))
    print("Cross Validation Score Decision Tree: %.2f%%" % (Cross_val_score * 100.0))
    print("Precision Score Decision Tree: %.2f%%" % (Precision_score * 100.0))
    print("Recall Score Decision Tree: %.2f%%" % (Recall_score * 100.0))
    print(classification_report(Y_test, Y_pred, target_names=target_names))
    print(confusion_matrix(Y_test, Y_pred))
    # print(X_train.shape)
    # print(X_train_new.shape)
    # print(model.feature_importances_)

def train_extra_tree(X,Y):
    # split data into train and test sets
    seed = 7
    test_size = 0.25
    target_names = ['ev','car']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    # fit model no training data
    model = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    # model = SelectFromModel(model)
    # model = RFE(model, 5, step=1)
    model.fit(X_train, Y_train)
    joblib.dump(model, '/home/abhishek/Documents/projects/RAVEV/ravev/weights/feature_based/ravev_extra_tree.pkl')
    scores = cross_val_score(model, X_train, Y_train)
    Cross_val_score = scores.mean()

    # make predictions for test data
    Y_pred = model.predict(X_test)
    predictions = [round(value) for value in Y_pred]

    # model = SelectFromModel(model, prefit=True)
    # X_train_new = model.transform(X_train)

    # evaluate predictions
    accuracy = accuracy_score(Y_test, predictions)
    Precision_score = precision_score(Y_test, predictions)
    Recall_score = recall_score(Y_test, predictions)
    print("Accuracy Extra Tree: %.2f%%" % (accuracy * 100.0))
    print("Cross Validation Score Extra Tree: %.2f%%" % (Cross_val_score * 100.0))
    print("Precision Score Extra Tree: %.2f%%" % (Precision_score * 100.0))
    print("Recall Score Extra Tree: %.2f%%" % (Recall_score * 100.0))
    print(classification_report(Y_test, Y_pred, target_names=target_names))
    print(confusion_matrix(Y_test, Y_pred))
    # print(X_train.shape)
    # print(X_train_new.shape)
    # print(model.feature_importances_)

def train_adaboost(X,Y):
    # split data into train and test sets
    seed = 7
    test_size = 0.25
    target_names = ['ev','car']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    # fit model no training data
    model = AdaBoostClassifier(n_estimators=100)
    # model = SelectFromModel(model)
    # model = RFE(model, 5, step=1)
    model.fit(X_train, Y_train)
    joblib.dump(model, '/home/abhishek/Documents/projects/RAVEV/ravev/weights/feature_based/ravev_adaboost.pkl')
    scores = cross_val_score(model, X_train, Y_train)
    Cross_val_score = scores.mean()
  
    # make predictions for test data
    Y_pred = model.predict(X_test)
    predictions = [round(value) for value in Y_pred]

    # model = SelectFromModel(model, prefit=True)
    # X_train_new = model.transform(X_train) 

    # evaluate predictions
    accuracy = accuracy_score(Y_test, predictions)
    Precision_score = precision_score(Y_test, predictions)
    Recall_score = recall_score(Y_test, predictions)
    print("Accuracy Adaboost: %.2f%%" % (accuracy * 100.0))
    print("Cross Validation Score Adaboost: %.2f%%" % (Cross_val_score * 100.0))
    print("Precision Score Adaboost: %.2f%%" % (Precision_score * 100.0))
    print("Recall Score Adaboost: %.2f%%" % (Recall_score * 100.0))
    print(classification_report(Y_test, Y_pred, target_names=target_names))
    print(confusion_matrix(Y_test, Y_pred))
    # print(X_train.shape)
    # print(X_train_new.shape)
    # print(model.feature_importances_)

def train_gradient_boosting(X,Y):
        # split data into train and test sets
    seed = 7
    test_size = 0.25
    target_names = ['ev','car']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model no training data
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                     max_depth=1, random_state=0)
    # model = SelectFromModel(model)
    # model = RFE(model, 5, step=1)
    model.fit(X_train, Y_train)
    joblib.dump(model, '/home/abhishek/Documents/projects/RAVEV/ravev/weights/feature_based/ravev_gradient_boosting.pkl')
    scores = cross_val_score(model, X_train, Y_train)
    Cross_val_score = scores.mean()
  
    # make predictions for test data
    Y_pred = model.predict(X_test)
    predictions = [round(value) for value in Y_pred]

    # model = SelectFromModel(model, prefit=True)
    # X_train_new = model.transform(X_train) 

    # evaluate predictions
    accuracy = accuracy_score(Y_test, predictions)
    Precision_score = precision_score(Y_test, predictions)
    Recall_score = recall_score(Y_test, predictions)
    print("Accuracy Gradient Boosting: %.2f%%" % (accuracy * 100.0))
    print("Cross Validation Score Gradient Boosting: %.2f%%" % (Cross_val_score * 100.0))
    print("Precision Score Gradient Boosting: %.2f%%" % (Precision_score * 100.0))
    print("Recall Score Gradient Boosting: %.2f%%" % (Recall_score * 100.0))
    print(classification_report(Y_test, Y_pred, target_names=target_names))
    print(confusion_matrix(Y_test, Y_pred))
    # print(X_train.shape)
    # print(X_train_new.shape)
    # print(model.feature_importances_)

def train_random_forrest(X,Y):
    # split data into train and test sets
    seed = 7
    test_size = 0.25
    target_names = ['ev','car']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    # fit model no training data
    model = RandomForestClassifier(n_estimators=10, max_depth=None,
                                 min_samples_split=2, random_state=0)                             
    # model = SelectFromModel(model)
    # model = RFE(model, 5, step=1)
    model.fit(X_train, Y_train)
    joblib.dump(model, '/home/abhishek/Documents/projects/RAVEV/ravev/weights/feature_based/ravev_random_forrest.pkl')
    scores = cross_val_score(model, X_train, Y_train)
    Cross_val_score = scores.mean()
  
    # make predictions for test data
    Y_pred = model.predict(X_test)
    predictions = [round(value) for value in Y_pred]

    # model = SelectFromModel(model, prefit=True)
    # X_train_new = model.transform(X_train)

    # evaluate predictions
    accuracy = accuracy_score(Y_test, predictions)
    Precision_score = precision_score(Y_test, predictions)
    Recall_score = recall_score(Y_test, predictions)
    print("Accuracy Random Forrest: %.2f%%" % (accuracy * 100.0))
    print("Cross Validation Score Random Forrest: %.2f%%" % (Cross_val_score * 100.0))
    print("Precision Score Random Forrest: %.2f%%" % (Precision_score * 100.0))
    print("Recall Score Random Forrest: %.2f%%" % (Recall_score * 100.0))
    print(classification_report(Y_test, Y_pred, target_names=target_names))
    print(confusion_matrix(Y_test, Y_pred))
    # print(X_train.shape)
    # print(X_train_new.shape)
    # print(model.feature_importances_)


def train_kNeighbours(X, Y):

    # split data into train and test sets
    seed = 7
    test_size = 0.25
    target_names = ['ev','car']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    # fit model no training data
    model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                 metric='minkowski', metric_params=None, n_jobs=1)
    # model = SelectFromModel(model)
    # model = RFE(model, 5, step=1)
    model.fit(X_train, Y_train)
    joblib.dump(model, '/home/abhishek/Documents/projects/RAVEV/ravev/weights/feature_based/ravev_k_neighbours.pkl')
    scores = cross_val_score(model, X_train, Y_train)
    Cross_val_score = scores.mean()
  
    # make predictions for test data
    Y_pred = model.predict(X_test)
    predictions = [round(value) for value in Y_pred]

    # model = SelectFromModel(model, prefit=True)
    # X_train_new = model.transform(X_train)

    # evaluate predictions
    accuracy = accuracy_score(Y_test, predictions)
    Precision_score = precision_score(Y_test, predictions)
    Recall_score = recall_score(Y_test, predictions)
    print("Accuracy K Neighbours: %.2f%%" % (accuracy * 100.0))
    print("Cross Validation Score K Neighbours: %.2f%%" % (Cross_val_score * 100.0))
    print("Precision Score K Neighbours: %.2f%%" % (Precision_score * 100.0))
    print("Recall Score K Neighbours: %.2f%%" % (Recall_score * 100.0))
    print(classification_report(Y_test, Y_pred, target_names=target_names))
    print(confusion_matrix(Y_test, Y_pred))
    # print(X_train.shape)
    # print(X_train_new.shape)
    # print(model.feature_importances_)  


def ravev_metrics():
    print 1
    #print(confusion_matrix(y_test, Y_pred))  
    # print(classification_report(y_test, Y_pred))

def temp_predict():
    print 1

if __name__ == '__main__':
    main()
