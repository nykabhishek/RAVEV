from imutils import paths
import cv2
#import nltk
import numpy as np
import joblib
from skimage import feature
import os,glob,imutils,argparse
#from matplotlib import pyplot as plt
from sklearn import svm, neighbors
# from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier 
from sklearn.preprocessing import label_binarize, LabelEncoder

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="dataset file path")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for knn classification")
args = vars(ap.parse_args())


def main():
    print("[INFO] handling images...")
    imagePaths = list(paths.list_images(args["dataset"]))
    rawImages, hist_features, hog_descriptors, labels = image_feature_arrays(imagePaths)
    # show some information on the memory consumed by the raw images
    # matrix and hist_features matrix
    rawImages = np.array(rawImages)
    hist_features = np.array(hist_features)
    hog_descriptors = np.array(hog_descriptors)
    labels = np.array(labels)
    print("Image Pixel matrix: {:.2f}MB".format(rawImages.nbytes / (1024 * 1000.0)))
    print("Color Histogram feature matrix: {:.2f}MB".format(hist_features.nbytes / (1024 * 1000.0)))
    print("HOG Feature matrix: {:.2f}MB".format(hog_descriptors.nbytes / (1024 * 1000.0)))

    # train_kNeighbours(rawImages, hist_features, hog_descriptors, labels)
    # train_mlp_neural_network(rawImages, hist_features, hog_descriptors, labels)
    # train_svm(rawImages, hist_features, hog_descriptors, labels)
    # train_adaboost(rawImages, hist_features, hog_descriptors, labels)
    # train_gradient_boosting(rawImages, hist_features, hog_descriptors, labels)
    # train_random_forrest(rawImages, hist_features, hog_descriptors, labels)
    # train_XGboost(rawImages, hist_features, hog_descriptors, labels)
    # train_decision_tree(rawImages, hist_features, hog_descriptors, labels)
    # train_extra_tree(rawImages, hist_features, hog_descriptors, labels)
    tabulate_results(rawImages, hist_features, hog_descriptors, labels)

def ravev_predict(image, algorithm):
    if algorithm=='KNN':
        img_weights = joblib.load('weights/image_based/KNN'+str(args["neighbors"])+'/KNN'+str(args["neighbors"])+'_RawImage.pkl')
        img_pred = img_weights.predict(image)
        hist_weights = joblib.load('weights/image_based/KNN'+str(args["neighbors"])+'/KNN'+str(args["neighbors"])+'_Histogram.pkl')
        hist_pred = hist_weights.predict(image)
        hog_weights = joblib.load('weights/image_based/KNN'+str(args["neighbors"])+'/KNN'+str(args["neighbors"])+'_HOG.pkl')
        hog_pred = hog_weights.predict(image)  
        print("[INFO] predictions from k-nn classifier with k=%d neighbours: \n \
                RawImage based prediction - %d \n \
                Histogram based predictions - %d \n \
                HOG based predictions - %d" % (args["neighbors"], img_pred, hist_pred, hog_pred) )
    else:
        img_weights = joblib.load('weights/image_based/'+str(algorithm)+'/'+str(algorithm)+'_RawImage.pkl')
        img_pred = img_weights.predict(image)
        hist_weights = joblib.load('weights/image_based/'+str(algorithm)+'/'+str(algorithm)+'_Histogram.pkl')
        hist_pred = hist_weights.predict(image)
        hog_weights = joblib.load('weights/image_based/'+str(algorithm)+'/'+str(algorithm)+'_HOG.pkl')
        hog_pred = hog_weights.predict(image)  
        print("[INFO] predictions from "+str(algorithm)+" classifier: \n \
                RawImage based prediction - %d \n \
                Histogram based predictions - %d \n \
                HOG based predictions - %d" % (img_pred, hist_pred, hog_pred) )
    return img_pred, hist_pred, hog_pred

def image_to_pixel_vector(image, size=(128, 128)):
    '''resize the image to a fixed size, then flatten the image into an array of raw pixel intensities'''
    return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(32, 32, 32)):
    '''extract a 3D color histogram from the HSV color space using the specified number of `bins` per channel'''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
        [0, 256, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3
    else:
        cv2.normalize(hist, hist)

    ''''return the flattened histogram as the feature vector'''
    return hist.flatten()

def hog_features(image, size=(128, 128)):
    img = cv2.resize(image, size)

    hog = feature.hog(img, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    
    return hog.flatten()


def image_feature_arrays(imagePaths):

    rawImages = []
    hist_features = []
    hog_descriptors = []
    labels = []
    
    for (i, imagePath) in enumerate(imagePaths):
        # load the image and extract the class label
        # our images were named as labels.image_number.format
        image = cv2.imread(imagePath)
        # get the labels from the name of the images by extract the string before "."
        label = imagePath.split(os.path.sep)[-1].split("_")[0]

        # extract raw pixel intensity "features"
        #followed by a color histogram to characterize the color distribution of the pixels
        # in the image
        pixels = image_to_pixel_vector(image)
        hog = hog_features(image)
        hist = extract_color_histogram(image)

        # add the messages we got to the raw images, features, and labels matricies
        rawImages.append(pixels)
        hist_features.append(hist)
        hog_descriptors.append(hog)
        labels.append(label)

        # show an update every 200 images until the last image
        if i > 0 and ((i + 1)% 200 == 0 or i ==len(imagePaths)-1):
            print("[INFO] processed {}/{}".format(i+1, len(imagePaths)))

    return rawImages, hist_features, hog_descriptors, labels


def train_kNeighbours(rawImages, hist_features, hog_descriptors, labels):

    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages, labels, test_size=0.15, random_state=42)
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
        hist_features, labels, test_size=0.15, random_state=42)
    (trainHOG, testHOG, trainHOGlabels, testHOGlabels) = train_test_split(
        hog_descriptors, labels, test_size=0.15, random_state=42)

    # k-NN
    print("\n")
    print("[INFO] Evaluating raw image pixel accuracy...")
    model = KNeighborsClassifier(n_neighbors=args["neighbors"])
    model.fit(trainRI, trainRL)
    joblib.dump(model, 'weights/image_based/KNN'+str(args["neighbors"])+'/KNN'+str(args["neighbors"])+'_RawImage.pkl', compress=True)
    acc = model.score(testRI, testRL)
    print("[INFO] k-NN classifier: k=%d" % args["neighbors"])
    print("[INFO] raw image pixel accuracy: {:.2f}%".format(acc * 100))

    print("\n")
    print("[INFO] Evaluating histogram accuracy...")
    model = KNeighborsClassifier(n_neighbors=args["neighbors"])
    model.fit(trainFeat, trainLabels)
    joblib.dump(model, 'weights/image_based/knn'+str(args["neighbors"])+'/knn'+str(args["neighbors"])+'_Histogram.pkl', compress=True)
    acc = model.score(testFeat, testLabels)
    print("[INFO] k-NN classifier: k=%d" % args["neighbors"])
    print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))
    
    # k-NN
    print("\n")
    print("[INFO] Evaluating HOG accuracy...")
    model = KNeighborsClassifier(n_neighbors=args["neighbors"])
    model.fit(trainHOG, trainHOGlabels)
    joblib.dump(model, 'weights/image_based/KNN'+str(args["neighbors"])+'/KNN'+str(args["neighbors"])+'_HOG.pkl', compress=True)
    acc = model.score(testHOG, testHOGlabels)
    print("[INFO] k-NN classifier: k=%d" % args["neighbors"])
    print("[INFO] HOG accuracy: {:.2f}%".format(acc * 100))


def train_mlp_neural_network(rawImages, hist_features, hog_descriptors, labels):

    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages, labels, test_size=0.15, random_state=42)
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
        hist_features, labels, test_size=0.15, random_state=42)
    (trainHOG, testHOG, trainHOGlabels, testHOGlabels) = train_test_split(
        hog_descriptors, labels, test_size=0.15, random_state=42)

   # MLP neural network
    print("\n")
    print("[INFO] Evaluating raw image pixel accuracy...")
    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=1e-4,
                        solver='sgd', tol=1e-4, random_state=1,
                        learning_rate_init=.1)
    model.fit(trainRI, trainRL)
    joblib.dump(model, 'weights/image_based/MLP/MLP_RawImage.pkl', compress=True)
    acc = model.score(testRI, testRL)
    print("[INFO] neural network raw image pixel accuracy: {:.2f}%".format(acc * 100))


    # MLP neural network
    print("\n")
    print("[INFO] Evaluating histogram accuracy...")
    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=1e-4,
                        solver='sgd', tol=1e-4, random_state=1,
                        learning_rate_init=.1)
    model.fit(trainFeat, trainLabels)
    joblib.dump(model, 'weights/image_based/MLP/MLP_Histogram.pkl', compress=True)
    acc = model.score(testFeat, testLabels)
    print("[INFO] neural network histogram accuracy: {:.2f}%".format(acc * 100))

    # MLP neural network
    print("\n")
    print("[INFO] Evaluating HOG accuracy...")
    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=1e-4,
                        solver='sgd', tol=1e-4, random_state=1,
                        learning_rate_init=.1)
    model.fit(trainHOG, trainHOGlabels)
    joblib.dump(model, 'weights/image_based/MLP/MLP_HOG.pkl', compress=True)
    acc = model.score(testHOG, testHOGlabels)
    print("[INFO] neural network HOG accuracy: {:.2f}%".format(acc * 100))


def train_svm(rawImages, hist_features, hog_descriptors, labels):

    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages, labels, test_size=0.15, random_state=42)
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
        hist_features, labels, test_size=0.15, random_state=42)
    (trainHOG, testHOG, trainHOGlabels, testHOGlabels) = train_test_split(
        hog_descriptors, labels, test_size=0.15, random_state=42)

    #SVC
    print("\n")
    print("[INFO] Evaluating raw image pixel accuracy...")
    model = svm.SVC(max_iter=5000,class_weight='balanced')
    model.fit(trainRI, trainRL)
    joblib.dump(model, 'weights/image_based/SVM/SVM_RawImage.pkl', compress=True)
    acc = model.score(testRI, testRL)
    print("[INFO] SVM-SVC raw image pixel accuracy: {:.2f}%".format(acc * 100))


    #SVC
    print("\n")
    print("[INFO] Evaluating histogram accuracy...")
    model = svm.SVC(max_iter=5000,class_weight='balanced')
    model.fit(trainFeat, trainLabels)
    joblib.dump(model, 'weights/image_based/SVM/SVM_Histogram.pkl', compress=True)
    acc = model.score(testFeat, testLabels)
    print("[INFO] SVM-SVC histogram accuracy: {:.2f}%".format(acc * 100))

    #SVC
    print("\n")
    print("[INFO] Evaluating HOG accuracy...")
    model = svm.SVC(max_iter=5000,class_weight='balanced')
    model.fit(trainHOG, trainHOGlabels)
    joblib.dump(model, 'weights/image_based/SVM/SVM_HOG.pkl', compress=True)
    acc = model.score(testHOG, testHOGlabels)
    print("[INFO] SVM-SVC HOG accuracy: {:.2f}%".format(acc * 100))

def train_adaboost(rawImages, hist_features, hog_descriptors, labels):


    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages, labels, test_size=0.15, random_state=42)
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
        hist_features, labels, test_size=0.15, random_state=42)
    (trainHOG, testHOG, trainHOGlabels, testHOGlabels) = train_test_split(
        hog_descriptors, labels, test_size=0.15, random_state=42)

    #Adaboost
    print("\n")
    print("[INFO] Evaluating raw image pixel accuracy...")
    model = AdaBoostClassifier(n_estimators=100)
    model.fit(trainRI, trainRL)
    joblib.dump(model, 'weights/image_based/Adaboost/Adaboost_RawImage.pkl', compress=True)
    acc = model.score(testRI, testRL)
    print("[INFO] Adaboost raw image pixel accuracy: {:.2f}%".format(acc * 100))


    #Adaboost
    print("\n")
    print("[INFO] Evaluating histogram accuracy...")
    model = AdaBoostClassifier(n_estimators=100)
    model.fit(trainFeat, trainLabels)
    joblib.dump(model, 'weights/image_based/Adaboost/Adaboost_Histogram.pkl', compress=True)
    acc = model.score(testFeat, testLabels)
    print("[INFO] Adaboost histogram accuracy: {:.2f}%".format(acc * 100))

    #Adaboost
    print("\n")
    print("[INFO] Evaluating HOG accuracy...")
    model = AdaBoostClassifier(n_estimators=100)
    model.fit(trainHOG, trainHOGlabels)
    joblib.dump(model, 'weights/image_based/Adaboost/Adaboost_HOG.pkl', compress=True)
    acc = model.score(testHOG, testHOGlabels)
    print("[INFO] Adaboost HOG accuracy: {:.2f}%".format(acc * 100))

def train_gradient_boosting(rawImages, hist_features, hog_descriptors, labels):


    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages, labels, test_size=0.15, random_state=42)
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
        hist_features, labels, test_size=0.15, random_state=42)
    (trainHOG, testHOG, trainHOGlabels, testHOGlabels) = train_test_split(
        hog_descriptors, labels, test_size=0.15, random_state=42)

    #Gradient_Boosting
    print("\n")
    print("[INFO] Evaluating raw image pixel accuracy...")
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                        max_depth=1, random_state=0)
    model.fit(trainRI, trainRL)
    joblib.dump(model, 'weights/image_based/gradient_boosting/GradientBoosting_RawImage.pkl', compress=True)
    acc = model.score(testRI, testRL)
    print("[INFO] Gradient Boosting raw image pixel accuracy: {:.2f}%".format(acc * 100))


    #Gradient_Boosting
    print("\n")
    print("[INFO] Evaluating histogram accuracy...")
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                        max_depth=1, random_state=0)
    model.fit(trainFeat, trainLabels)
    joblib.dump(model, 'weights/image_based/gradient_boosting/GradientBoosting_Histogram.pkl', compress=True)
    acc = model.score(testFeat, testLabels)
    print("[INFO] Gradient Boosting histogram accuracy: {:.2f}%".format(acc * 100))

    #Gradient Boosting
    print("\n")
    print("[INFO] Evaluating HOG accuracy...")
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                        max_depth=1, random_state=0)
    model.fit(trainHOG, trainHOGlabels)
    joblib.dump(model, 'weights/image_based/gradient_boosting/GradientBoosting_HOG.pkl', compress=True)
    acc = model.score(testHOG, testHOGlabels)
    print("[INFO] Gradient Boosting HOG accuracy: {:.2f}%".format(acc * 100))


def train_random_forrest(rawImages, hist_features, hog_descriptors, labels):


    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages, labels, test_size=0.15, random_state=42)
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
        hist_features, labels, test_size=0.15, random_state=42)
    (trainHOG, testHOG, trainHOGlabels, testHOGlabels) = train_test_split(
        hog_descriptors, labels, test_size=0.15, random_state=42)

    #Random_Forest
    print("\n")
    print("[INFO] Evaluating raw image pixel accuracy...")
    model = RandomForestClassifier(n_estimators=10, max_depth=None,
                                    min_samples_split=2, random_state=0)
    model.fit(trainRI, trainRL)
    joblib.dump(model, 'weights/image_based/random_forrest/RandomForrest_RawImage.pkl', compress=True)
    acc = model.score(testRI, testRL)
    print("[INFO] Random Forest raw image pixel accuracy: {:.2f}%".format(acc * 100))


    #Random_Forest
    print("\n")
    print("[INFO] Evaluating histogram accuracy...")
    model = RandomForestClassifier(n_estimators=10, max_depth=None,
                                    min_samples_split=2, random_state=0)
    model.fit(trainFeat, trainLabels)
    joblib.dump(model, 'weights/image_based/random_forrest/RandomForrest_Histogram.pkl', compress=True)
    acc = model.score(testFeat, testLabels)
    print("[INFO] Random Forest histogram accuracy: {:.2f}%".format(acc * 100))

    #Random_Forest
    print("\n")
    print("[INFO] Evaluating HOG accuracy...")
    model = RandomForestClassifier(n_estimators=10, max_depth=None,
                                    min_samples_split=2, random_state=0)
    model.fit(trainHOG, trainHOGlabels)
    joblib.dump(model, 'weights/image_based/random_forrest/RandomForrest_HOG.pkl', compress=True)
    acc = model.score(testHOG, testHOGlabels)
    print("[INFO] Random Forest HOG accuracy: {:.2f}%".format(acc * 100))

def train_XGboost(rawImages, hist_features, hog_descriptors, labels):


    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages, labels, test_size=0.15, random_state=42)
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
        hist_features, labels, test_size=0.15, random_state=42)
    (trainHOG, testHOG, trainHOGlabels, testHOGlabels) = train_test_split(
        hog_descriptors, labels, test_size=0.15, random_state=42)

    #XGboost
    print("\n")
    print("[INFO] Evaluating raw image pixel accuracy...")
    model = XGBClassifier()
    model.fit(trainRI, trainRL)
    joblib.dump(model, 'weights/image_based/XGboost/XGboost_RawImage.pkl', compress=True)
    acc = model.score(testRI, testRL)
    print("[INFO] XGboost raw image pixel accuracy: {:.2f}%".format(acc * 100))


    #XGboost
    print("\n")
    print("[INFO] Evaluating histogram accuracy...")
    model = XGBClassifier()
    model.fit(trainFeat, trainLabels)
    joblib.dump(model, 'weights/image_based/XGboost/XGboost_Histogram.pkl', compress=True)
    acc = model.score(testFeat, testLabels)
    print("[INFO] XGboost histogram accuracy: {:.2f}%".format(acc * 100))

    #XGboost
    print("\n")
    print("[INFO] Evaluating HOG accuracy...")
    model = XGBClassifier()
    model.fit(trainHOG, trainHOGlabels)
    joblib.dump(model, 'weights/image_based/XGboost/XGboost_HOG.pkl', compress=True)
    acc = model.score(testHOG, testHOGlabels)
    print("[INFO] XGboost HOG accuracy: {:.2f}%".format(acc * 100))


def train_decision_tree(rawImages, hist_features, hog_descriptors, labels):


    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages, labels, test_size=0.15, random_state=42)
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
        hist_features, labels, test_size=0.15, random_state=42)
    (trainHOG, testHOG, trainHOGlabels, testHOGlabels) = train_test_split(
        hog_descriptors, labels, test_size=0.15, random_state=42)

    #Decision_Tree
    print("\n")
    print("[INFO] Evaluating raw image pixel accuracy...")
    model = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    model.fit(trainRI, trainRL)
    joblib.dump(model, 'weights/image_based/decision_tree/DecisionTree_RawImage.pkl', compress=True)
    acc = model.score(testRI, testRL)
    print("[INFO] Decision_Tree raw image pixel accuracy: {:.2f}%".format(acc * 100))


    #Decision_Tree
    print("\n")
    print("[INFO] Evaluating histogram accuracy...")
    model = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    model.fit(trainFeat, trainLabels)
    joblib.dump(model, 'weights/image_based/decision_tree/DecisionTree_Histogram.pkl', compress=True)
    acc = model.score(testFeat, testLabels)
    print("[INFO] Decision_Tree histogram accuracy: {:.2f}%".format(acc * 100))

    #Decision_Tree
    print("\n")
    print("[INFO] Evaluating HOG accuracy...")
    model = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    model.fit(trainHOG, trainHOGlabels)
    joblib.dump(model, 'weights/image_based/decision_tree/DecisionTree_HOG.pkl', compress=True)
    acc = model.score(testHOG, testHOGlabels)
    print("[INFO] Decision_Tree HOG accuracy: {:.2f}%".format(acc * 100))


def train_extra_tree(rawImages, hist_features, hog_descriptors, labels):

    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages, labels, test_size=0.15, random_state=42)
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
        hist_features, labels, test_size=0.15, random_state=42)
    (trainHOG, testHOG, trainHOGlabels, testHOGlabels) = train_test_split(
        hog_descriptors, labels, test_size=0.15, random_state=42)

    #Extra_Tree
    print("\n")
    print("[INFO] Evaluating raw image pixel accuracy...")
    model = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    model.fit(trainRI, trainRL)
    joblib.dump(model, 'weights/image_based/extra_tree/ExtraTree_RawImage.pkl', compress=True)
    acc = model.score(testRI, testRL)
    print("[INFO] Extra_Tree raw image pixel accuracy: {:.2f}%".format(acc * 100))

    #Extra_Tree
    print("\n")
    print("[INFO] Evaluating histogram accuracy...")
    model = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    model.fit(trainFeat, trainLabels)
    joblib.dump(model, 'weights/image_based/extra_tree/ExtraTree_Histogram.pkl', compress=True)
    acc = model.score(testFeat, testLabels)
    print("[INFO] Extra_Tree histogram accuracy: {:.2f}%".format(acc * 100))

    #Extra_Tree
    print("\n")
    print("[INFO] Evaluating HOG accuracy...")
    model = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    model.fit(trainHOG, trainHOGlabels)
    joblib.dump(model, 'weights/image_based/extra_tree/ExtraTree_HOG.pkl', compress=True)
    acc = model.score(testHOG, testHOGlabels)
    print("[INFO] Extra_Tree HOG accuracy: {:.2f}%".format(acc * 100))

def tabulate_results(rawImages, hist_features, hog_descriptors, labels):
    encoded_column_vector = label_binarize(labels, classes=['car','ev']) # ham will be 0 and spam will be 1
    encoded_labels = np.ravel(encoded_column_vector)

    root = "/home/abhishek/Documents/projects/RAVEV/ravev/weights/image_based"    
    for path, subdirs, files in os.walk(root):
        for name in files:
            algorithm = name.split("_")[0]
            extn = name.split("_")[1]
            clf_feature = extn.split(".")[0]
            file_name = os.path.join(path, name)
            print algorithm, clf_feature

            if clf_feature=="RawImage":
                (train_data, test_data, train_label, test_label) = train_test_split(
                    rawImages, encoded_labels, test_size=0.15, random_state=42)
                result_features(train_data, test_data, train_label, test_label, algorithm, clf_feature)
            elif clf_feature=="Histogram":
                (train_data, test_data, train_label, test_label) = train_test_split(
                    hist_features, encoded_labels, test_size=0.15, random_state=42)
                result_features(train_data, test_data, train_label, test_label, algorithm, clf_feature)
            elif clf_feature=="HOG":
                (train_data, test_data, train_label, test_label) = train_test_split(
                    hog_descriptors, encoded_labels, test_size=0.15, random_state=42)
                result_features(train_data, test_data, train_label, test_label, algorithm, clf_feature)


    

def result_features(train_data, test_data, train_label, test_label, algorithm, clf_feature):
    le = LabelEncoder()
    le.fit(["ev","car"])
    target_names = ['ev','car']
    model = joblib.load('weights/image_based/'+str(algorithm)+'/'+str(algorithm)+'_'+str(clf_feature)+'.pkl')
    predicted_label = model.predict(test_data)
    predictions = [round(value) for value in le.transform(predicted_label)]
    # scores = cross_val_score(model, train_data, train_label)
    # Cross_val_score = scores.mean()
    accuracy = accuracy_score(test_label, predictions)
    Precision_score = precision_score(test_label, predictions)
    Recall_score = recall_score(test_label, predictions)
    print("\nAccuracy %s - %s : %.2f%%" % (str(algorithm),str(clf_feature),(accuracy * 100.0)))
    # print("Cross Validation Score %s - %s: %.2f%%" % (str(algorithm),str(clf_feature),(Cross_val_score * 100.0)))
    print("\nPrecision Score %s - %s: %.2f%%" % (str(algorithm),str(clf_feature),(Precision_score * 100.0)))
    print("\nRecall Score %s - %s: %.2f%%" % (str(algorithm),str(clf_feature),(Recall_score * 100.0)))
    print("\nClassification Report \n")
    print(classification_report(test_label, predictions, target_names=target_names))
    print("Confusion Matrix")
    print(confusion_matrix(test_label, predictions))


if __name__ == '__main__':
    main()
