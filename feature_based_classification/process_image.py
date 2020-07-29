import cv2
import numpy as np

def feature_extract(image, x_min , x_max, y_min, y_max, detclass):

        [height, width, channels] = image.shape
        # height = y_max - y_min
        # width = x_max - x_min
        # image_area = height*width

        
        x_mid = height/2
        y_max_20 = 0.2*(height)
 
        roi_bgr = image[int(y_min):int(y_max) , int(x_min):int(x_max)]
        roi_bgr_t20 = image[int(y_min):int(y_max_20) , int(x_min):int(x_max)]
        #print line_num

        # # Convert BGR to HSV
        # roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        # roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # try:
            ##**** Explain Method 1 & 2 ****##
        total_area_hsv_red, max_contour_area_hsv_red, centroid_X_hsv_red, centroid_Y_hsv_red = find_contour_features('red',roi_bgr,'1')
        total_area_hsv_blue, max_contour_area_hsv_blue, centroid_X_hsv_blue, centroid_Y_hsv_blue = find_contour_features('blue',roi_bgr,'1') 

        total_area_hsv_red_t20, max_contour_area_hsv_red_t20, centroid_X_hsv_red_t20, centroid_Y_hsv_red_t20 = find_contour_features('red',roi_bgr_t20,'1')
        total_area_hsv_blue_t20, max_contour_area_hsv_blue_t20, centroid_X_hsv_blue_t20, centroid_Y_hsv_blue_t20 = find_contour_features('blue',roi_bgr_t20,'1')


        'binary thresholding'
        total_area_hsv_thresh_red, max_contour_area_hsv_thresh_red, centroid_X_hsv_thresh_red, centroid_Y_hsv_thresh_red = find_contour_features('red',roi_bgr,'2')
        total_area_hsv_thresh_blue, max_contour_area_hsv_thresh_blue, centroid_X_hsv_thresh_blue, centroid_Y_hsv_thresh_blue = find_contour_features('blue',roi_bgr,'2') 

        total_area_hsv_thresh_red_t20, max_contour_area_hsv_thresh_red_t20, centroid_X_hsv_thresh_red_t20, centroid_Y_hsv_thresh_red_t20 = find_contour_features('red',roi_bgr_t20,'2')
        total_area_hsv_thresh_blue_t20, max_contour_area_hsv_thresh_blue_t20, centroid_X_hsv_thresh_blue_t20, centroid_Y_hsv_thresh_blue_t20 = find_contour_features('blue',roi_bgr_t20,'2')

        # except cv2.error:
        #     continue

        feature_array = [[  float(total_area_hsv_blue), float(total_area_hsv_red), \
                            float(max_contour_area_hsv_red), float(max_contour_area_hsv_blue), \
                            float(centroid_X_hsv_red), float(centroid_Y_hsv_red), \
                            float(centroid_X_hsv_blue), float(centroid_Y_hsv_blue), \
                            float(total_area_hsv_blue_t20), float(total_area_hsv_red_t20), \
                            float(max_contour_area_hsv_red_t20), float(max_contour_area_hsv_blue_t20), \
                            float(centroid_X_hsv_red_t20), float(centroid_Y_hsv_red_t20), \
                            float(centroid_X_hsv_blue_t20), float(centroid_Y_hsv_blue_t20), \
                            float(total_area_hsv_thresh_blue), float(total_area_hsv_thresh_red), \
                            float(max_contour_area_hsv_thresh_red), float(max_contour_area_hsv_thresh_blue), \
                            float(centroid_X_hsv_thresh_red), float(centroid_Y_hsv_thresh_red), \
                            float(centroid_X_hsv_thresh_blue), float(centroid_Y_hsv_thresh_blue), \
                            float(total_area_hsv_thresh_blue_t20), float(total_area_hsv_thresh_red_t20), \
                            float(max_contour_area_hsv_thresh_red_t20), float(max_contour_area_hsv_thresh_blue_t20), \
                            float(centroid_X_hsv_thresh_red_t20), float(centroid_Y_hsv_thresh_red_t20), \
                            float(centroid_X_hsv_thresh_blue_t20), float(centroid_Y_hsv_thresh_blue_t20)   ]]

        return feature_array

def find_contour_features(color, image, method):
       
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