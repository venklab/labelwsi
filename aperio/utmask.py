#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import zlib
from math import sqrt

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as cocomask

CV2_MAJOR_VERSION = int(cv2.__version__.split('.')[0])


def calc_dist(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def masklocation(mask):
    return masklocation_v2(mask)


def masklocation_v2(mask):
    """ find:
        - bbox: in xmin, xmax, ymin, ymax order
        - distance to border, in top, right, bottom, left order(same as CSS)
    The input shall be a 2D array, 0 for backgroud

    masklocation_v2 is 10 times faster than v1, 0.3ms vs 3ms for a test mask
    """
    # empty mask
    if np.count_nonzero(mask) == 0:
        return {}

    maskx = np.any(mask, axis=0)
    masky = np.any(mask, axis=1)
    x1 = np.argmax(maskx)
    y1 = np.argmax(masky)
    h, w = mask.shape
    x2 = w - np.argmax(maskx[::-1])
    y2 = h - np.argmax(masky[::-1])

    #if x1 == 0 and x2 == w and y1 == 0 and y2 == h:
    #    return {}

    return {'bbox': (x1, x2, y1, y2),
            'dist': (y1, w - x2, h - y2, x1)}


def masklocation_v1(mask):
    """ find:
        - bbox: in xmin, xmax, ymin, ymax order
        - distance to border, in top, right, bottom, left order(same as CSS)
    The input shall be a 2D array, 0 for backgroud """

    pos = mask.nonzero()
    if len(pos[0]) == 0:  # empty mask
        return {}

    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    h, w = mask.shape
    return {'bbox': (xmin, xmax, ymin, ymax),
            'dist': (ymin, w - xmax, h - ymax, xmin)}


def mask2crop(mask):
    """ crop the input full size mask image to the border of mask, the input
    shall be a 2D array, 0 for backgroup """

    loc = masklocation(mask)
    if not loc:  # sometimes the mask is empty
        print('mask2crop: mask is empty')
        return {}

    xmin, xmax, ymin, ymax = loc['bbox']
    crop = mask[ymin:(ymax+1), xmin:(xmax+1)]
    area = np.count_nonzero(crop)
    blob = mask2rleblob(crop)
    h, w = crop.shape
    return {'x': int(xmin), 'y': int(ymin), 'w': w, 'h': h,
            'blob': blob, 'mask': crop, 'area': area}


def mask2shape(mask, mpp):
    """ convert mask to polygon, returned coordinates is [(x, y), .. ] """
    # cv.CHAIN_APPROX_TC89_KCOS has least points on edge, compare to
    # CHAIN_APPROX_TC89_L1 and CHAIN_APPROX_SIMPLE
    """
    # opencv 4.x
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
        #cv2.CHAIN_APPROX_TC89_KCOS)
        #cv2.CHAIN_APPROX_TC89_L1)
    # opencv 3.x
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_TC89_KCOS)
        #cv2.CHAIN_APPROX_SIMPLE)
        #cv2.CHAIN_APPROX_TC89_L1)
    """
    if CV2_MAJOR_VERSION == 4:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_TC89_KCOS)
    else:
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_TC89_KCOS)
    points = None
    shapes = []
    for cnt in contours:
        total = cnt.shape[0]
        points = np.reshape(cnt, (total, 2))
        count = len(points)
        shapes.append((count, points))

    shapes.sort(key=lambda x: x[0])  # choose the shape with most edge points
    s = shapes[-1][1]
    npoints = s.shape[0]
    i = 0
    delete_list = []
    # delete dense points
    while i < npoints:
        j = i + 1
        while j < npoints:
            d = calc_dist(s[i], s[j]) * mpp  # in um
            if d > 2:
                break
            delete_list.append(j)
            j += 1
        i = j

    s2 = np.delete(s, delete_list, axis=0)
    return s2


def mask2rleblob(mask):
    """
    size of few random tile blobs:

    colorpng=8057, graypng=2550, rle=1044(zlib=510, lzma=588, bz2=567)
    colorpng=9108, graypng=3373, rle=1558(zlib=793, lzma=868, bz2=854)
    colorpng=8296, graypng=2700, rle=1416(zlib=783, lzma=816, bz2=881)

    binary mask, rle then zlib compress results in smallest blob

    pycoco rle counts is further encoded with a varient of LEB128, but using
    6bits/char and ascii chars 48-111 (maskApi.c)
    """
    rle = cocomask.encode(np.asfortranarray(mask))
    # {'size': [1024, 1024], 'counts': b'xxxx...'} = rle
    rle['counts'] = rle['counts'].decode('ascii')
    blob = zlib.compress(json.dumps(rle).encode('ascii'), level=9)
    return blob


def rleblob2mask(blob):
    """
    rle blob is the zlib compressed rle['counts']
    Return a np array
    """
    rle = json.loads(zlib.decompress(blob).decode('ascii'))
    rle['counts'] = rle['counts'].encode('ascii')  # expect bytes like
    mask = cocomask.decode(rle)
    return mask.astype(bool)
