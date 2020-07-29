import numpy as np
import pandas as pd
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt


# Create arbitrary dataset for example
#X = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),]
df = pd.DataFrame()

df = pd.read_excel('excel/train_output.xlsx')

df = df[[   'i', 'line', 'img_area', 'Class', 'Class_id', 'x_min', 'x_max', 'y_min', 'y_max', \
            'total_area_hsv_blue', 'total_area_hsv_red', \
            'max_contour_area_hsv_red', 'max_contour_area_hsv_blue', \
            'centroid_X_hsv_red', 'centroid_Y_hsv_red', \
            'centroid_X_hsv_blue', 'centroid_Y_hsv_blue', \
            'total_area_hsv_blue_t20', 'total_area_hsv_red_t20', \
            'max_contour_area_hsv_red_t20', 'max_contour_area_hsv_blue_t20', \
            'centroid_X_hsv_red_t20', 'centroid_Y_hsv_red_t20', \
            'centroid_X_hsv_blue_t20', 'centroid_Y_hsv_blue_t20', \
            'total_area_hsv_thresh_blue', 'total_area_hsv_thresh_red', \
            'max_contour_area_hsv_thresh_red', 'max_contour_area_hsv_thresh_blue', \
            'centroid_X_hsv_thresh_red', 'centroid_Y_hsv_thresh_red', \
            'centroid_X_hsv_thresh_blue', 'centroid_Y_hsv_thresh_blue', \
            'total_area_hsv_thresh_blue_t20', 'total_area_hsv_thresh_red_t20', \
            'max_contour_area_hsv_thresh_red_t20', 'max_contour_area_hsv_thresh_blue_t20', \
            'centroid_X_hsv_thresh_red_t20', 'centroid_Y_hsv_thresh_red_t20', \
            'centroid_X_hsv_thresh_blue_t20', 'centroid_Y_hsv_thresh_blue_t20']]



# Fit Support Vector Machine Classifier

y = df['Class_id']
fit_type = 'rbf'

fig = plt.figure()

plt.subplot(2, 4, 1)
X_1 = df[['total_area_hsv_blue', 'total_area_hsv_red']]
clf = svm.SVC(kernel=fit_type, decision_function_shape='ovr')
clf.fit(X_1.values, y.values)
plot_decision_regions(X=X_1.values, 
                      y=y.values,
                      clf=clf, 
                      legend=2)       
plt.xlabel(X_1.columns[0], size=8)
plt.ylabel(X_1.columns[1], size=8)
plt.title('SVM Decision Region Boundary_1', size=10)


plt.subplot(2, 4, 2)
X_2 = df[['max_contour_area_hsv_red', 'max_contour_area_hsv_blue']]
clf = svm.SVC(kernel=fit_type, decision_function_shape='ovr')
clf.fit(X_2.values, y.values)
plot_decision_regions(X=X_2.values, 
                      y=y.values,
                      clf=clf, 
                      legend=2)      
plt.xlabel(X_2.columns[0], size=8)
plt.ylabel(X_2.columns[1], size=8)
plt.title('SVM Decision Region Boundary_2', size=10)

plt.subplot(2, 4, 3)
X_3 = df[['total_area_hsv_blue_t20', 'total_area_hsv_red_t20']]
clf = svm.SVC(kernel=fit_type, decision_function_shape='ovr')
clf.fit(X_3.values, y.values)
plot_decision_regions(X=X_3.values, 
                      y=y.values,
                      clf=clf, 
                      legend=2)       
plt.xlabel(X_3.columns[0], size=8)
plt.ylabel(X_3.columns[1], size=8)
plt.title('SVM Decision Region Boundary_3', size=10)

plt.subplot(2, 4, 4)
X_4 = df[['max_contour_area_hsv_red_t20', 'max_contour_area_hsv_blue_t20']]
clf = svm.SVC(kernel=fit_type, decision_function_shape='ovr')
clf.fit(X_4.values, y.values)
plot_decision_regions(X=X_4.values, 
                      y=y.values,
                      clf=clf, 
                      legend=2)       
plt.xlabel(X_4.columns[0], size=8)
plt.ylabel(X_4.columns[1], size=8)
plt.title('SVM Decision Region Boundary_4', size=10)

plt.subplot(2, 4, 5)
X_5 = df[['total_area_hsv_thresh_blue', 'total_area_hsv_thresh_red']]
clf = svm.SVC(kernel=fit_type, decision_function_shape='ovr')
clf.fit(X_5.values, y.values)
plot_decision_regions(X=X_5.values, 
                      y=y.values,
                      clf=clf, 
                      legend=2)       
plt.xlabel(X_5.columns[0], size=8)
plt.ylabel(X_5.columns[1], size=8)
plt.title('SVM Decision Region Boundary_5', size=10)

plt.subplot(2, 4, 6)
X_6 = df[['max_contour_area_hsv_thresh_red', 'max_contour_area_hsv_thresh_blue']]
clf = svm.SVC(kernel=fit_type, decision_function_shape='ovr')
clf.fit(X_6.values, y.values)
plot_decision_regions(X=X_6.values, 
                      y=y.values,
                      clf=clf, 
                      legend=2)       
plt.xlabel(X_6.columns[0], size=8)
plt.ylabel(X_6.columns[1], size=8)
plt.title('SVM Decision Region Boundary_6', size=10)

plt.subplot(2, 4, 7)
X_7 = df[['total_area_hsv_thresh_blue_t20', 'total_area_hsv_thresh_red_t20']]
clf = svm.SVC(kernel=fit_type, decision_function_shape='ovr')
clf.fit(X_7.values, y.values)
plot_decision_regions(X=X_7.values, 
                      y=y.values,
                      clf=clf, 
                      legend=2)       
plt.xlabel(X_7.columns[0], size=8)
plt.ylabel(X_7.columns[1], size=8)
plt.title('SVM Decision Region Boundary_7', size=10)

plt.subplot(2, 4, 8)
X_8 = df[['max_contour_area_hsv_thresh_red_t20', 'max_contour_area_hsv_thresh_blue_t20']]
clf = svm.SVC(kernel=fit_type, decision_function_shape='ovr')
clf.fit(X_5.values, y.values)
plot_decision_regions(X=X_8.values, 
                      y=y.values,
                      clf=clf, 
                      legend=2)       
plt.xlabel(X_8.columns[0], size=8)
plt.ylabel(X_8.columns[1], size=8)
plt.title('SVM Decision Region Boundary_8', size=10)


plt.show()