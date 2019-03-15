from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time
import cv2
import numpy as np
import os
from os import listdir
from skimage.feature import greycomatrix, greycoprops
from sklearn.preprocessing import normalize
from contour import *
import pickle

# start = time.time()

#output
class_0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #ulmus carpinifolia
class_1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #acer
class_2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #salix aurita
class_3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #quercus
class_4 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #alnus incana
class_5 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] #betula pubescens
class_6 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] #salix alba 'sericea'
class_7 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] #populus tremula
class_8 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] #ulmus glabra
class_9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] #sorbus aucuparia
class_10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] #salix sinerea
class_11 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] #populus
class_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] #tilia
class_13 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] #sorbus intermedia
class_14 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] #fagus silvatica
array_class = [class_0, class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10, class_11, class_12, class_13, class_14]

def toVector(base_img):
    check = []
    img_ori = cv2.imread(base_img)
    
    #---------------preprocessing---------------
    #otsu thresholding
    img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    ret, img_th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, img_th = cv2.threshold(img_th,0,255,cv2.THRESH_BINARY_INV)
    #morphology closing        
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) #strel disk
    img_closed = cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, strel)
    #morphology opening
    img_opened = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, strel)
    #floodfill / bwareaopen
    im_floodfill = img_opened.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = img_opened.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # Invert floodfilled image
    img_filled_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    img_out = img_opened | img_filled_inv
    img_out_rgb = cv2.cvtColor(img_out,cv2.COLOR_GRAY2RGB)
    leaf = img_out_rgb & img_ori
    
#    path = '../../preprocessed/leaf'+str(digit)+'/'
#    cv2.imwrite(path+file,leaf)
    #---------------preprocessing---------------
            
    #---------------8 shape features---------------
    i, contours, hierarchy = cv2.findContours(img_out,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    check.append(contours)
    cnt = contours[0]
    c = Contour(img,cnt)
    area = c.area
    perimeter = c.perimeter
    metric = 4*3.14*area/(pow(perimeter,2))
    eccentricity = c.eccentricity
    convexarea = c.convex_area
    majoraxislength = c.majoraxis_length
    minoraxislength = c.minoraxis_length
    solidity = c.solidity
    shape_features = np.array([area, perimeter, metric, eccentricity, convexarea, majoraxislength, minoraxislength, solidity], float)
    shape_features = shape_features.tolist()
    #---------------shape features---------------
            
    #---------------72 glcm features---------------
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['energy', 'homogeneity','dissimilarity','contrast','correlation','ASM']
    glcm = greycomatrix(img, distances=distances, angles=angles, symmetric=True, normed=True)
    glcm_features = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
    glcm_features = glcm_features.tolist()
    #---------------glcm features---------------
    
    all_features = np.array(shape_features + glcm_features,float)

    return all_features

def doPredict(data):
#------------load min and max data from training data features
	lines = tuple(open('min_max_glcm.txt', 'r'))
	min_max = []
	for x in lines:
		split_text = x.split(' ')
		min_max.append([split_text[0],split_text[1][:-2]])
	for x in min_max:
		x[0] = float(x[0])
		x[1] = float(x[1])
#------------load min and max data from training data features
        
#-------normalization
	for i in range(data.shape[0]):
		data[i] = (data[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])
#-------normalization
    
	data = data.reshape(1, -1) #needed for 1 sample prediction
	filename = 'finalized_model.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	loaded_model.fit
	result = loaded_model.predict(data)

	return result

#normalization
# for i in range(features.shape[1]):
#     column = features[:,i]
#     min_val = column.min()
#     max_val = column.max()
#     for j in range(features.shape[0]):
#         features[j][i] = (features[j][i] - min_val) / (max_val - min_val)


# train_images, test_images, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

# start_training = time.time()

# #train
# print("Training..")
# mlp = MLPClassifier(hidden_layer_sizes=(30), max_iter=1, alpha=1e-4, activation='logistic',
#                     solver='adam', verbose=10, tol=1e-4, random_state=1,
#                     learning_rate_init=0.01, warm_start=True)

# for i in range(1):
#     mlp.fit(train_images, train_labels)

# print("Training set score: %f" % mlp.score(train_images, train_labels))
# print("Test set score: %f" % mlp.score(test_images, test_labels))

# end = time.time()
# print("Time to train: " + str(end - start_training))
# print("Time for all: " + str(end - start))
# filename = 'finalized_model.sav'
# pickle.dump(mlp, open(filename, 'wb'))

# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(test_images, test_labels)
# print(result)

#references
#https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
#https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html
#https://stackoverflow.com/questions/17251541/disk-structuring-element-opencv-vs-matlab
#https://stackoverflow.com/questions/18339988/implementing-imcloseim-se-in-opencv\
#https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
#http://opencvpython.blogspot.com/2012/04/contour-features.html

#cv2.imwrite( "../tes.jpg", img_out);