#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from io import BytesIO, StringIO
from PIL import Image, ImageDraw


# blob in PNG format
def narray2pngblob(arr):
    stream = BytesIO()
    tmp = Image.fromarray(arr)
    tmp.save(stream, format='PNG')
    blob = stream.getvalue()
    return blob


def pngblob2narray(blob):
    stream = BytesIO(blob)
    img = Image.open(stream)
    return np.asarray(img)


# blob in JPEG format
def narray2jpgblob(arr):
    stream = BytesIO()
    tmp = Image.fromarray(arr)
    tmp.save(stream, format='JPEG')
    blob = stream.getvalue()
    return blob


def jpgblob2narray(blob):
    return pngblob2narray(blob)


# blob in NP format
def narray2blob(arr):
    fp = BytesIO()
    np.save(fp, arr)
    fp.seek(0)
    return fp.read()

def blob2narray(blob):
    fp = BytesIO(blob)
    fp.seek(0)
    return np.load(fp)


def array2pngfp(arr):
    """ create a file object from input ndarray """
    img = Image.fromarray(arr)
    fp = BytesIO()
    img.save(fp, format='PNG')
    fp.seek(0)
    return fp


def img2pngfp(img):
    """ create a file object from input ndarray """
    fp = BytesIO()
    img.save(fp, format='PNG')
    fp.seek(0)
    return fp


def _put_one_mask_in_region(cvsm, newv, mx, my, mask, w, h):
    """ fast enough, usually use less than 0.0001 second """

    mh, mw = mask.shape
    mx1 = mx + mw - 1
    my1 = my + mh - 1
    mcrop_x = 0
    mcrop_x1 = mw
    needcrop = False
    cvsd_x, cvsd_x1, cvsd_y, cvsd_y1 = mx, mx1, my, my1
    if mx <= 0:
        mcrop_x = -1 * mx
        cvsd_x = 0
        needcrop = True

    if mx1 >= (w - 1):
        mcrop_x1 = mw - (mx1 - (w - 1) + 1)
        cvsd_x1 = w
        needcrop = True

    mcrop_y = 0
    mcrop_y1 = mh
    if my <= 0:
        mcrop_y = -1 * my
        cvsd_y = 0
        needcrop = True

    if my1 >= (h - 1):
        mcrop_y1 = mh - (my1 - (h - 1) + 1)
        cvsd_y1 = h - 1
        needcrop = True

    if needcrop:
        np.copyto(cvsm[cvsd_y:cvsd_y1+1, cvsd_x:cvsd_x1+1], newv,
                  where=mask[mcrop_y:mcrop_y1+1, mcrop_x:mcrop_x1+1])

    else:
        np.copyto(cvsm[my:my1+1, mx:mx1+1], newv, where=mask)

    return cvsm

