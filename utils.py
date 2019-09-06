import getpass
import socket

import core
import constants
import requests
import cv2
import os
import numpy as np
#from web.client import ContextClassifierClient
from matplotlib import pyplot as plt

import pickle
import uuid
import warnings
import os
import glob
import itertools


def get_tso_candidates(screen, normed=False):
    # if screen.context is None: warnings.warn("Finding tso candidates of screen whose context is None")

    text_boxes = find_text_box(screen)
    mser = cv2.MSER_create()
    img = screen.img_bgr()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    regions, _ = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    # cv2.polylines(vis, hulls, 1, (0, 255, 0))

    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    tso_candidates = []
    for contour in hulls:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 0.80 * img.shape[0] * img.shape[1]: continue
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    #     x,y,w,h = cv2.boundingRect(contour)
    #     cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2)

    #     text_only = cv2.bitwise_and(img, img, mask=mask)

    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))

    _, cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        #     cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
        x, y, w, h = cv2.boundingRect(contour)
        if normed:
            img_h, img_w = screen.img.shape[:2]
            img_h, img_w = float(img_h), float(img_w)
            x, y, w, h = x / img_w, y / img_h, w / img_w, h / img_h
        bbox = core.BoundingBox(x, y, w, h)

        if not is_bbox_in_text_boxes(bbox, text_boxes):
            # cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
            tso = core.TransitionSourceObject(None, screen.context, bbox)
            tso_candidates.append(tso)

    return tso_candidates


def find_text_box(screen):
    img = screen.img
    img_id = uuid.uuid1()
    img_name = str(img_id) + ".jpg"
    img_path = os.path.join(constants.DIR_SCREEN, img_name)
    plt.imsave(img_path, img)
    text_boxes_resp = get_text_boxes(img_path)
    text_boxes = []
    for box in text_boxes_resp:
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        bbox = core.BoundingBox(x, y, w, h)
        text_boxes.append(bbox)
    return text_boxes


def get_text_boxes(screen_path):
    payload = {"screen": open(screen_path, 'rb')}
    response = requests.request("POST", constants.OCR_URL, files=payload)
    bboxes = response.json()
    return bboxes['text_boxes']


def is_bbox_in_text_boxes(bbox, text_boxes):
    bbox_center = bbox.centre
    for text_box in text_boxes:
        if bbox_center in text_box:
            return True
    return False


def img_crop(img, bbox):
    return img[bbox.y:bbox.y + bbox.h, bbox.x:bbox.x + bbox.w, ...]




