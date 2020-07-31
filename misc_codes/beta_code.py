import numpy as np
import argparse
import cv2
import os
from imutils import paths
from skimage import exposure
from skimage import feature
from smic import SMIC


def main():
    hog_view_images()

def smic():
    clf = SMIC()
    clf.prepare_train_data('/home/Documents/projects/RAVEV/smic_data')
    hyperparameters = clf.search_optimal_parameters()
    #hyperparameters = {'transfer_model' : 'vgg16', 'optimizer' : 'sgd', 
	#				'top_layers' : [['dense', 512, 'relu'],['dense', 512, 'relu']]}
    clf.fit(hyperparameters, epochs = 50, batch_size=32)

def prepare_dataset():

    #file_list = glob.glob('/*.TXT')
    file_num = 584
    jobtype = "train" # "train" or "test"


    for i in range(0,file_num):
        imgfile = cv2.imread('/home/Documents/projects/RAVEV/dataset/'+str(jobtype)+'/test'+str(i)+'.jpg')
        txtfile = open('/home/Documents/projects/RAVEV/dataset/'+str(jobtype)+'/test'+str(i)+'.txt','r+')

        [height, width, channels] = imgfile.shape

        line_num=0
        for line in txtfile.readlines():
            line_num += 1
            [Class_id, x, y, w, h] = line.split(' ')
            x_min = int( (float(width)*float(x)) - ((float(width)*float(w))/2) )
            y_min = int( float(height)*float(y) - ((float(height)*float(h))/2) )
            x_max = x_min + int(float(width)*float(w))
            y_max = y_min + int(float(height)*float(h))
            y_max_20 = y_min + 0.2*(y_max - y_min)
            #print x_min, x_max, y_min, y_max
            roi_bgr = imgfile[int(y_min):int(y_max) , int(x_min):int(x_max)]

            if Class_id == '1':
                cv2.imwrite('/home/Documents/projects/RAVEV/dataset/train/ev/ev_'+str(i)+'_'+str(line_num)+'.jpg',roi_bgr)
                Class = 'EV'
            elif Class_id =='2':
                cv2.imwrite('/home/Documents/projects/RAVEV/dataset/train/car/car_'+str(i)+'_'+str(line_num)+'.jpg',roi_bgr)
                Class = 'car'


def resize_view_images():
    images = []
    for filename in os.listdir("../../dataset"):
        image = cv2.imread(os.path.join("../../dataset",filename))
        image = cv2.resize(image, (100, 100))
        while True:
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.imshow('frame',image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def hog_view_images():
    images = []
    for (i, imagePath) in enumerate(paths.list_images("../../dataset")):
        image = cv2.imread(imagePath)
        res_image = cv2.resize(image, (100, 100))
        (H, hogImage) = feature.hog(res_image, orientations=9, pixels_per_cell=(10, 10),
		                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualise=True)
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")
        while True:
            cv2.imshow("HOG Image #{}".format(i + 1), hogImage)
            if cv2.waitKey(1)   & 0xFF == ord('q'):
                break



if __name__ == '__main__':
    main()
