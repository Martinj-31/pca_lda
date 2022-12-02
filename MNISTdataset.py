# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 16:52:00 2022

@author: Mingyu
"""

import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' %kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' %kind)

    with open(labels_path, 'rb') as lbpath:
        # > : Big endian
        # I : unsigned integer
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels