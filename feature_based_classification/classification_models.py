import os,glob
import cv2
import joblib
import numpy as np
import process_image

class FeatureClassifier(object):

    def __init__(self,  model='adaboost'):
            super(FeatureClassifier, self).__init__()

            self.classifier_dict =  {'svm':'weights/feature_based/ravev_svm.pkl',\
                                'XGBoost':'weights/feature_based/ravev_XGBoost.pkl',\
                                'decision':'weights/feature_based/ravev_decision_tree.pkl',\
                                'extra':'weights/feature_based/ravev_extra_tree.pkl',\
                                'adaboost':'weights/feature_based/ravev_adaboost.pkl',\
                                'gradient':'weights/feature_based/ravev_gradient_boosting.pkl',\
                                'random':'weights/feature_based/ravev_random_forrest.pkl',\
                                'knn':'weights/feature_based/ravev_k_neighbours.pkl'}
            self.model = self.classifier_dict[model]

    def predict(self, image, xmin , xmax, ymin, ymax, detclass):
        feature_array = process_image.feature_extract(image, xmin , xmax, ymin, ymax, detclass)
        # clf = joblib.load(self.model)
        # Y_pred = clf.predict(feature_array)
        # print Y_pred
        # return Y_pred
        return feature_array
        

def main():
    node = FeatureClassifier()


if __name__ == '__main__':
    main()