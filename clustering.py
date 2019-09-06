import cv2
import numpy as np
import os
import core
import pickle
import warnings
import os
import glob
import itertools
from PIL import  Image
import matplotlib.pyplot as plt
import uuid
import constants
import requests
import json
from tqdm import tqdm
from keras.preprocessing import image
#from keras.applications.vgg16 import VGG16
#from keras.applications.vgg16 import preprocess_input
#from keras.applications.mobilenet_v2 import MobileNetV2
#from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from utils import get_tso_candidates, find_text_box, get_text_boxes, is_bbox_in_text_boxes, img_crop

model = InceptionV3(weights='imagenet', include_top=False)


def get_bboxes(img):
    tsos = get_tso_candidates(core.Screen(img))
    return [t.bbox for t in tsos]



def save_cropped_images(folder_path, destination_path):
    count=0
    for image in tqdm(os.listdir(folder_path)):
        screen = plt.imread(folder_path + "/"+image)
        bboxes = get_bboxes(screen)
        for bbox in bboxes:
            cropped_image = img_crop(screen,bbox)
            outpath = os.path.join(destination_path, "{}.jpg".format(uuid.uuid4()))
            Image.fromarray(cropped_image).save(outpath)

def find_object_clusters(image_path,save_icon_path,save_cluster_path):
    mobilenet_feature_list = []
    #image_path = "C:\\Users\\hgani\\train"
    save_cropped_images(image_path,save_icon_path)
    for images in os.listdir(save_icon_path):
        img = image.load_img(os.path.join(save_icon_path,images), target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        mobilenet_feature = model.predict(img_data)
        mobilenet_feature_np = np.array(mobilenet_feature)
        mobilenet_feature_list.append(mobilenet_feature_np.flatten())
        mobilenet_feature_list_np = np.array(mobilenet_feature_list)
    #kmeans = KMeans(n_clusters=50, random_state=0).fit(mobilenet_feature_list_np)
    dbscan = DBSCAN(eps=0.5,metric='euclidean', min_samples=2).fit(mobilenet_feature_list_np)
    y_kmeans = dbscan.labels_
    #print(y_kmeans)
    max_labels = max(y_kmeans)
    for i in range(0,len(y_kmeans)):
        if y_kmeans[i] == -1:
            y_kmeans[i]= max_labels+1
    for i, j in zip(os.listdir(save_icon_path), y_kmeans):
        #print(i, j)
        if not os.path.exists(save_cluster_path):
            os.makedirs(save_cluster_path)

        #image2 = Image.open(os.path.join(save_icon_path, i))
        icon = cv2.imread(os.path.join(save_icon_path,i))
        if not os.path.exists(os.path.join(save_cluster_path, str(j))):
            os.makedirs(os.path.join(save_cluster_path,str(j)))
        cv2.imwrite(save_cluster_path+ '/'+str(j) +'/'+ str(i),icon)
       
find_object_clusters("D:/Shared/Screenshot/CP_images","D:/Shared/Screenshot/crop","D:/Shared/Screenshot/clusters")



