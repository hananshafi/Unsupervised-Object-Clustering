import uuid

import numpy as np
import matplotlib.pyplot as plt

import cv2

class Point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def __repr__(self):
        return "Point({}, {})".format(self.x, self.y)

class BoundingBox:

    def __init__(self, x, y, w, h):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)



    @classmethod
    def from_extremities(cls, xmin, ymin, xmax, ymax):
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        w = xmax - xmin
        h = ymax - ymin
        return cls(xmin, ymin, w, h)

    def __contains__(self, item):
        if not isinstance(item, Point):
            raise ValueError("Only instances of Point class can be queried")

        if (self.x <= item.x <= self.x + self.w) and (self.y <= item.y <= self.y + self.h):
            return True

        return False
    
    @property
    def centre(self):
        return Point(self.x + self.w/2, self.y + self.h/2)
      

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, dct):
        return cls(dct['x'], dct['y'], dct['w'], dct['h'])
    
    def __repr__(self):
        return "BoundingBox(x={},y={},w={},h={})".format(self.x, self.y, self.w, self.h)



class Screen:
    def __init__(self, img, context=None):
        self.img = img
        self.context = context

    @classmethod
    def from_file(cls, fpath):
        img = plt.imread(fpath)
        return cls(img)

    def set_context(self, context):
        self.context = context

    def img_bgr(self):
        return self.img[:,:,::-1]



class ScreenState(Screen):
    def __init__(self, img, context=None, tsos=None):
        super().__init__(img, context)
        self.tsos = tsos
        self.bboxes = [tso.bbox for tso in tsos]







