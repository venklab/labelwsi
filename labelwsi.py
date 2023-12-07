#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" labelsam: label image with sam """

import math
import os
import pickle
import sqlite3
import sys
import time

import cairo
import cv2
import numpy as np
import torch
import skimage.draw as ski_draw
from skimage.transform import resize as ski_resize
from skimage.transform import rescale as ski_rescale
from PIL import Image, ImageDraw
from deepzoom import DeepZoomGenerator

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib, Gio, GObject

from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, SamPredictor

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from aperio.utslide import UTSlide
from aperio.utslide import query_db
from aperio.utslide import mask2rleblob, rleblob2mask
from aperio.utslide import mask2crop, masklocation
import utimage

from wsicommon import narray2pngblob, pngblob2narray, narray2jpgblob, \
        jpgblob2narray, narray2blob, blob2narray

sam_checkpoint = '/sw/download/checkpoint/sam_vit_h_4b8939.pth'
model_type = 'vit_h'
model = None


WIN_X, WIN_Y = 1700, 1100
WIN_X, WIN_Y = 1700, 1000  # in x2go
#WIN_X, WIN_Y = 1300, 700
MAX_SCALE = 2
MASK_MIN_SCALE = 0.4
START_SCALE = 0.5  # initial scale, also the min scale
DELAY_UPDATE = 0.3  # time waiting for update after mouse zoom
BIRDVIEW_W, BIRDVIEW_H = 200, 200
CR_LEFT, CR_BOTTOM = 100, 40  # CR_LEFT is the width of left panel
RIGHT_HAND, LEFT_HAND = True, False
#TILE_SIZE = 1022
#TILE_OVERLAP = 1  # TILE_SIZE + 2 * overlap should be power of 2
TILE_SIZE = 510
TILE_OVERLAP = 1

APPDIR = os.path.abspath(os.path.dirname(sys.argv[0]))

GLB_DEBUG = True
GLB_RNDCOLOR = False

reuse_pickle = False

GREEN, YELLOW = [0, 255, 0, 127], [255, 255, 0, 170]
CYAN = [0, 255, 255, 127]
TOMATO = [255, 99, 71, 200]
ORANGE = [255, 165, 0, 200]
BLUE = [0, 0, 255, 200]
RED = [255, 0, 0, 170]


def dprint(msg, *args):
    if GLB_DEBUG:
        print('debug: ' + msg % args)


def ut_load_sam_predictor():
    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'

    print('Loading check point from %s' % sam_checkpoint)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    print('    check point loaded')

    sam.to(device=device)
    return SamPredictor(sam)


# Deep zoom
def get_tile_address(x, y):
    """ x,y is on current deep zoom level, return (col, row) """
    col = int(x / TILE_SIZE)
    row = int(y / TILE_SIZE)
    return (col, row)


def dzxy_to_tilexy(x, y):
    x, y = int(x), int(y)
    col, row = get_tile_address(x, y)  # on top level deepzoom
    offsetx = 0 if col == 0 else TILE_OVERLAP
    offsety = 0 if row == 0 else TILE_OVERLAP
    tx = x % TILE_SIZE
    ty = y % TILE_SIZE
    return (tx, ty)


def get_mask_tile_address(dzslide, x, y, w, h):
    """ find tiles cooresponding to a region (x,y,w,h) on slide level0

    return a list of (dzlevel, c1, c2, r1, r2), start with top deepzoom level
    """
    res = []
    r = 1.0
    dzlevel = dzslide.level_count - 1
    while dzlevel >= dzslide.minlevel:
        c1, r1 = get_tile_address(x * r, y * r)
        c2, r2 = get_tile_address((x+w-1)*r, (y+h-1)*r)
        res.append((dzlevel, c1, c2, r1, r2))
        dzlevel -= 1
        r *= 0.5

    return res


def get_or_set_tile(db, dzslide, level, address):
    """ return: (tile, commit_flag), the tile is a narray """
    col, row = address
    sql = 'select tile from SlideTile WHERE lid=? AND cid=? AND rid=?'
    needcommit = False
    """
    ck = '%d_%d_%d' % (level, col, row)
    if ck in dzslide.tile_cache:
        return (dzslide.tile_cache[ck], needcommit)
    """

    qrow = query_db(db, sql, (level, col, row), one=True)
    if qrow:
        arr = jpgblob2narray(qrow[0])
        return (arr, needcommit)

    # origin get_tile in deepzoom takes 0.2s to read a 1024x1024 tile, it
    # actually reads 2048x2048 then scale down to 1024. After modify deepzoom
    # to only read 1024x1024, time reduced to 0.04s
    tile = dzslide.get_tile(level, (col, row))
    arr = np.asarray(tile)
    if dzslide.display_gamma is not None:
        arr = utimage.gamma_correct(arr, dzslide.display_gamma)

    blob = narray2jpgblob(arr)

    if level < dzslide.minlevel:
        dzslide.minlevel = level
        dprint('set dzslide.minlevel to %d' % dzslide.minlevel)

    sql_add = 'insert into SlideTile (lid,cid,rid,tile) VALUES (?,?,?,?)'
    db.execute(sql_add, (level, col, row, blob))
    needcommit = True
    return (arr,needcommit)


def surface_from_pil(im, alpha=1.0, format=cairo.FORMAT_ARGB32):
    """
    :param im: Pillow Image
    :param alpha: 0..1 alpha to add to non-alpha images
    :param format: Pixel format for output surface
    """
    assert format in (cairo.FORMAT_RGB24, cairo.FORMAT_ARGB32), "Unsupported pixel format: %s" % format
    if 'A' not in im.getbands():
        im.putalpha(int(alpha * 256.))
    arr = bytearray(im.tobytes('raw', 'BGRa'))
    surface = cairo.ImageSurface.create_for_data(arr, format, im.width, im.height)
    return surface


def resize_keep_ratio(im, outw, outh):
    """ resize an image to fit outw, outh """

    h, w = float(im.shape[0]),float(im.shape[1])
    multichannel = False
    if len(im.shape) > 2:
        multichannel = True

    s1 = outw / w
    s2 = outh / h

    im_resized = ski_rescale(im, min(s1, s2), preserve_range=True,
            multichannel=multichannel)

    return im_resized.astype(np.uint8)


def rndcolorimg_from_masks(masks, mids, changed=[], auto_bad=[],
                           manual_bad=[], edited=[], ignores=[]):
    """ generate a random color image from masks. Its alpha channel is half
    transparent at masks, total transparent(0) at background """
    h, w = masks.shape
    cvs = np.zeros((h, w, 4), dtype=np.uint8)

    colors = [[0, 255, 0, 127], [0, 0, 255, 127],
            [0, 255, 255, 127], [255, 255, 0, 127], [255, 255, 100, 127],
            [80, 70, 180, 127], [250, 80, 190, 127], [245, 145, 50, 127],
            [70, 150, 250, 127], [50, 190, 190, 127], [190, 190, 50, 127]]

    color_orig = [0, 255, 0, 127]
    # red and reddish colors for manually edited mask
    color_changed = [
        [255, 0, 0, 157],
        [235, 0, 30, 157],
        [215, 0, 60, 157],
        [195, 0, 90, 157],
        [175, 0, 120, 157],
        [155, 0, 0, 157],
        [135, 0, 60, 157],
            ]

    auto_bad = set(auto_bad)
    manual_bad = set(manual_bad)
    cmids = set(changed)
    edited = set(edited)
    # find unique is slow
    #u = np.unique(masks)
    for v in mids:
        if v == 0:
            continue
        if v in ignores:
            continue

        color = color_orig

        if v in auto_bad:
            color = YELLOW  # highlight bad mask
        elif v in manual_bad:
            color = TOMATO
        elif v in cmids:
            color = color_changed[v % 7] if GLB_RNDCOLOR else CYAN
        else:
            color = colors[v % 11] if GLB_RNDCOLOR else GREEN

        if v in edited:
            color = ORANGE  # highlight bad mask

        #cvs[masks == v] = color
        mask = masks == v
        np.copyto(cvs[:, :, 0], color[0], where=mask)
        np.copyto(cvs[:, :, 1], color[1], where=mask)
        np.copyto(cvs[:, :, 2], color[2], where=mask)
        np.copyto(cvs[:, :, 3], color[3], where=mask)

    return Image.fromarray(cvs)


def read_whole(slide):
    """
    read the entire level 0 image, return ndarray
    """
    z_size = (slide.width, slide.height)
    x = 0 + slide._l0_offset[0]
    y = 0 + slide._l0_offset[1]
    img = slide.read_region((x, y), 0, (slide.width, slide.height))
    img = img.convert('RGB')
    return np.asarray(img)


def gen_birdview(slide):
    z_size = (slide.width, slide.height)
    for i in range(slide.level_count):
        level = i
        lw, lh = slide.level_dimensions[level]
        if lw < BIRDVIEW_W and lh < BIRDVIEW_H:
            break

        #z_size = tuple(max(1, int(math.ceil(z / 2))) for z in z_size)

    x = 0 + slide._l0_offset[0]
    y = 0 + slide._l0_offset[1]
    img = slide.read_region((x, y), level, (lw, lh))
    if img.width > img.height:
        w = BIRDVIEW_W
        h = int(img.height * float(BIRDVIEW_W)/ img.width)
        upleft = (0, int((BIRDVIEW_H - h) * 0.5))
    else:
        h = BIRDVIEW_H
        w = int(img.width * float(BIRDVIEW_H)/ img.height)
        upleft = (int((BIRDVIEW_W - w) * 0.5), 0)

    img = img.resize((w, h))
    out = np.full((BIRDVIEW_H, BIRDVIEW_W, 4), 0, dtype=np.uint8)
    out[:,:,3] = 180
    outimg = Image.fromarray(out)
    outimg.paste(img, box=upleft)
    return (outimg, upleft)


def gen_crosshair():
    out = np.zeros((20, 20, 4), dtype=np.uint8)
    out[9:11, :, 1] = 200
    out[9:11, :, 3] = 255  # opaque

    out[:, 9:11, 1] = 200
    out[:, 9:11, 3] = 255
    return Image.fromarray(out)


def get_or_set_masktile(db, dzslide, dzlevel, address, needmask=False):
    col, row = address
    sql = 'select tile,fullsize_mask from MaskTile WHERE lid=? AND cid=? AND rid=?'
    qrow = query_db(db, sql, (dzlevel, col, row), one=True)
    needcommit = False
    if qrow:
        if needmask:
            masks_arr = pngblob2narray(qrow[1])
        else:
            masks_arr = None
        return (masks_arr, pngblob2narray(qrow[0]), needcommit)

    masks_arr, tile = _get_masktile(db, dzslide, dzlevel, (col, row))
    toplevel = dzslide.level_count - 1

    # narrayblob is 10x-40x bigger than png blob
    arr = np.asarray(tile)
    blobt = narray2pngblob(arr)

    if (dzslide.level_count - 1) == dzlevel:
        blobm = narray2pngblob(masks_arr)
    else:
        blobm = None

    sql_add = 'insert into MaskTile (lid,cid,rid,tile,fullsize_mask) VALUES (?,?,?,?,?)'
    db.execute(sql_add, (dzlevel, col, row, blobt, blobm))
    needcommit = True
    return (masks_arr, arr, needcommit)


def _get_masktile(db, dzslide, dzlevel, address):
    """ return the fullsize mask and scaled tile """
    col, row = address
    offsetx = 0 if col == 0 else TILE_OVERLAP
    offsety = 0 if row == 0 else TILE_OVERLAP
    # upper left of tile
    x = col * TILE_SIZE - offsetx
    y = row * TILE_SIZE - offsety
    w = TILE_SIZE + TILE_OVERLAP + offsetx
    h = TILE_SIZE + TILE_OVERLAP + offsety
    nlevels = dzslide.level_count
    dzscale = 2 ** (nlevels - 1 - dzlevel)

    # masks are uint16, store top dzlevel mask in db
    # get_masks_in_region_with_scale use 0.3s, so far most expensive. Time are
    # spent on resize boolean masks
    (masks_arr, mids, changed_mids, auto_bad, manual_bad,
        edited_mids) = get_masks_in_region_with_scale(db, dzscale, x, y, w, h)

    # rndcolorimg_from_masks use most of cpu when masks_arr is large,
    # however with smaller masks_arr, it use 0.01s
    img = rndcolorimg_from_masks(masks_arr, mids, changed_mids,
                                 auto_bad, manual_bad, edited_mids)
    return (masks_arr, img)


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
        #cvsm[cvsd_y:cvsd_y1+1, cvsd_x:cvsd_x1+1][
        #        mask[mcrop_y:mcrop_y1+1, mcrop_x:mcrop_x1+1]] = True
        np.copyto(cvsm[cvsd_y:cvsd_y1+1, cvsd_x:cvsd_x1+1], newv,
                  where=mask[mcrop_y:mcrop_y1+1, mcrop_x:mcrop_x1+1])

    else:
        #cvsm[my:my1+1, mx:mx1+1][mask] = True
        np.copyto(cvsm[my:my1+1, mx:mx1+1], newv, where=mask)

    return cvsm


def get_masks_in_region_with_scale(db, dzscale, lx, ly, lw, lh):
    """ a uint16 mask, value of mask is the db id """
    cvsm = np.zeros((lh, lw), dtype=np.uint16)
    sql = ('SELECT id,x,y,w,h,mask,polygon,is_bad,is_edit FROM Mask '
           ' WHERE ((x+w>? AND x+w<?) OR (x>? AND x<?) '
           '        OR (x<? AND x+w>?) OR (x<? AND x+w>?)) '
           '   AND ((y+h>? AND y+h<?) OR (y>? AND y<?) '
           '        OR (y<? AND y+h>?) OR (y<? AND y+h>?))')

    x = lx * dzscale
    y = ly * dzscale
    w = lw * dzscale
    h = lh * dzscale

    x1, y1 = x + w, y + h
    rows = query_db(db, sql, (x, x1, x, x1, x, x, x1, x1,
                              y, y1, y, y1, y, y, y1, y1))
    mids = []
    changed_mids = []
    edited_mids = []
    auto_bad_mids = []    # automatically marked bad mask
    manual_bad_mids = []  # marked as bad by human

    for m in rows:
        mid, mx, my, mw, mh, blob, blobp, is_bad, is_edit = m
        if blobp:
            changed_mids.append(mid)

        if is_bad == 1:
            auto_bad_mids.append(mid)
        elif is_bad == 2:
            manual_bad_mids.append(mid)

        if is_edit == 1:
            edited_mids.append(mid)

        w = int(float(mw) / dzscale)
        h = int(float(mh) / dzscale)
        mask = rleblob2mask(blob)
        # skimage.transform.resize is expensive, can take 0.02 to 0.05 s
        # utimage.resize_mask only resize boolean, take 1e-4 to 1e-3 s, speed
        # up 40x+
        #mask = resize_bool_mask(mask, w, h)
        mask = utimage.resize_mask(mask, w, h)

        mx = int(float(mx - x) / dzscale)
        my = int(float(my - y) / dzscale)
        _put_one_mask_in_region(cvsm, mid, mx, my, mask, lw, lh)

        mids.append(mid)

    return (cvsm, mids, changed_mids, auto_bad_mids, manual_bad_mids,
            edited_mids)


def resize_bool_mask(mask, w, h):
    # after resize, there are many non-zero pixels in background, shows
    # as vertical strips. Use a large value (100) to cut off these
    # resizing artifact
    tmp = np.zeros(mask.shape, dtype=np.uint8)
    tmp[mask] = 255
    tmp = ski_resize(tmp, (h, w), preserve_range=True)
    return tmp > 100


def get_masks_in_region(db, x, y, w, h):
    """ a uint16 mask, value of mask is the db id """
    cvsm = np.zeros((h, w), dtype=np.uint16)
    sql = ('SELECT id,x,y,w,h,mask,polygon FROM Mask '
           ' WHERE ((x+w>? AND x+w<?) OR (x>? AND x<?) '
           '        OR (x<? AND x+w>?) OR (x<? AND x+w>?)) '
           '   AND ((y+h>? AND y+h<?) OR (y>? AND y<?) '
           '        OR (y<? AND y+h>?) OR (y<? AND y+h>?))')
    x1, y1 = x + w, y + h
    rows = query_db(db, sql, (x, x1, x, x1, x, x, x1, x1,
                              y, y1, y, y1, y, y, y1, y1))
    mids = []
    changed_mids = []

    for m in rows:
        mid, mx, my, mw, mh, blob, blobp = m
        if blobp:
            changed_mids.append(mid)
        mask = rleblob2mask(blob)
        mx -= x
        my -= y

        _put_one_mask_in_region(cvsm, mid, mx, my, mask, w, h)

        mids.append(mid)

    return (cvsm, mids, changed_mids)


def draw_one_polygon(w, h, points):
    """ create a uint16 single channel image and draw a polygon, highlight each
    points with a small cycle, which value is the index in points array + 1
    Arguments:
        points: list of (x,y) tuple
    Return:
        narray
    """
    ymin, xmin = np.min(points, axis=0)
    ymax, xmax = np.max(points, axis=0)

    crop_x = crop_y = 0
    cvs_w, cvs_h = w, h
    if xmin < 0:
        cvs_w -= xmin
        crop_x = xmin * -1
        points += np.array([0, crop_x])

    if xmax > w:
        cvs_w += xmax - w

    if ymin < 0:
        cvs_h -= ymin
        crop_y = ymin * -1
        points += np.array([crop_y, 0])

    if ymax > h:
        cvs_h += ymax - h

    # add extra space for the point disks
    cvs_h += 10
    cvs_w += 10
    im = np.zeros((cvs_h, cvs_w), dtype=np.uint16)

    p0 = (points[0][0], points[0][1])
    for p in points[1:]:
        rr, cc, val = ski_draw.line_aa(p[0], p[1], p0[0], p0[1])
        im[rr, cc] = 65535 * val
        p0 = (p[0], p[1])

    # draw a disk at each point so that we can capture the point mouse click on
    # based on point value
    i = 1
    for p in points:
        if i == 1:
            rr, cc = ski_draw.disk(p, 9)
        else:
            rr, cc = ski_draw.disk(p, 6)
        im[rr, cc] = i  # disk value is the index of points + 1
        i += 1

    return im[crop_y:crop_y+h, crop_x:crop_x+w]


def mask2shape(mask, mpp):
    """ convert mask to polygon, returned coordinates is [(y, x), .. ] """
    # cv.CHAIN_APPROX_TC89_KCOS has least points on edge, compare to
    # CHAIN_APPROX_TC89_L1 and CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_TC89_KCOS)
        #cv2.CHAIN_APPROX_TC89_L1)
        #cv2.CHAIN_APPROX_SIMPLE)

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
    dprint('mpp = %.4f' % mpp)
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
    points_yx = [(p[1], p[0]) for p in s2]

    return points_yx


def calc_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def polygon2mask(points):
    """ create a uint8 single channel image and draw a polygon
    Arguments:
        points: list of (y,x) tuple
    Return:
        x, y, w, h, and the mask
        x, y is the upper left corner of mask
    """
    ymin, xmin = np.min(points, axis=0)
    ymax, xmax = np.max(points, axis=0)
    points = points.copy()
    points -= np.array([ymin, xmin])

    # add extra space for the point disks
    h = int(ymax - ymin + 1)
    w = int(xmax - xmin + 1)
    im = np.zeros((h, w), dtype=np.uint8)
    points_xy = [(p[1], p[0]) for p in points]
    img = Image.fromarray(im)
    draw = ImageDraw.Draw(img)
    draw.polygon(xy=points_xy, fill=1, outline=1)
    return (int(xmin), int(ymin), w, h, np.array(img, dtype=bool))


def get_polygon_direction(points):
    """ check whether from pn to pn+1 is right or left hand"""
    x,y,w,h,mask = polygon2mask(points)
    points = points.copy()
    points -= np.array([y, x])
    npoints = points.shape[0]
    p0, p0 = None, None
    i = 0
    while i < npoints-1:
        p0, p1 = points[i], points[i+1]
        # p0 p1 cannot be too close
        if calc_dist(p0, p1) > 10:
            break
        i += 1

    # draw a vector V from p0 to p1, let pm be the middle point, vector Vm is
    # from pm to pm1, length 2. Rotate Vm by 90 degrees, then check whether it
    # is inside polygon
    m = (p0 + p1) / 2  # middle point of p0 p1
    dp = p1 - p0
    lp = calc_dist(p0, p1)
    sinp = -1 * dp[0] / lp
    cosp = dp[1] / lp
    # move m to orig
    lm1 = 2
    m1x = lm1 * cosp
    m1y = lm1 * sinp   # y in cartesian
    # rotate 90 degrees
    m1x_p = -1.0 * m1y
    m1y_p = -1.0 * m1x  # y is back to screen coordinates
    # move m back to the middle of p0 and p1
    pm1 = np.array([m1y_p, m1x_p])
    pm1 += m
    t_x, t_y = int(pm1[1]), int(pm1[0])

    # add blank area around mask
    mask2 = np.zeros((h+10, w+10), dtype=np.uint8)
    mask2[5:5+h, 5:5+w] = mask
    if mask2[int(t_y+5), int(t_x+5)] != 0:
        # pn to pn+1 is the right hand direction
        pt_order = RIGHT_HAND
    else:
        pt_order = LEFT_HAND

    return pt_order


def update_db_schema(db):
    # update Mask table, add polygon column
    sql = ("SELECT count(*) FROM pragma_table_info('Mask') "
           " WHERE name='polygon'")
    row = query_db(db, sql, one=True)
    if row and row[0] == 0:
        db.execute('ALTER TABLE Mask ADD COLUMN polygon BLOB;')
        db.commit()

    # update Mask table, add is_edit column
    sql = ("SELECT count(*) FROM pragma_table_info('Mask') "
           " WHERE name='is_edit'")
    row = query_db(db, sql, one=True)
    if row and row[0] == 0:
        db.execute('ALTER TABLE Mask ADD COLUMN is_edit INTEGER DEFAULT 0;')
        db.commit()

    sql_add_table = """
CREATE TABLE IF NOT EXISTS SlideTile(
    id INTEGER PRIMARY KEY,
    lid INTEGER,
    rid   INTEGER,
    cid   INTEGER,
    tile BLOB
);

CREATE TABLE IF NOT EXISTS MaskTile(
    id INTEGER PRIMARY KEY,
    lid INTEGER,
    rid   INTEGER,
    cid   INTEGER,
    tile BLOB,
    fullsize_mask BLOB
);

CREATE INDEX IF NOT EXISTS idx_SlideTile_lid on SlideTile (lid);
CREATE INDEX IF NOT EXISTS idx_SlideTile_cid on SlideTile (cid);
CREATE INDEX IF NOT EXISTS idx_SlideTile_rid on SlideTile (rid);

CREATE INDEX IF NOT EXISTS idx_MaskTile_lid on MaskTile (lid);
CREATE INDEX IF NOT EXISTS idx_MaskTile_cid on MaskTile (cid);
CREATE INDEX IF NOT EXISTS idx_MaskTile_rid on MaskTile (rid);
"""
    db.executescript(sql_add_table)
    db.commit()


class Labelwsi(Gtk.Window):
    '''the main part'''

    def __init__(self, slide_path):
        super(Labelwsi, self).__init__()
        self.set_size_request(WIN_X, WIN_Y)

        self.predictor = None
        if not reuse_pickle:
            self.predictor = ut_load_sam_predictor()

        self.btn_box_sam = None
        self.load_slide(slide_path)
        self.init_gtk()

    def load_slide(self, slide_path):
        print('load %s' % slide_path)
        self.slide = UTSlide(slide_path)
        bname = os.path.basename(slide_path)
        self.set_title(bname)

        # self.slide.level_dimensions: list of dimensions, with index 0 be the
        # highest resolution, its lowest resolution may greater than (1, 1)
        # self.dzslide.level_dimensions has more dimensions, the resolution
        # goes down to (1, 1), which is at index 0, the last index has highest
        # resolution, it is the reverse of slide.level_dimensions

        update_db_schema(self.slide.db)

        # set limit_bounds for compensate Versa image x, y offset
        self.dzslide = DeepZoomGenerator(self.slide,
                tile_size=TILE_SIZE, overlap=TILE_OVERLAP,
                limit_bounds=True)
        self._set_dzslide_tile_min_level()
        self.dzslide.tile_cache = {}

        zeiss_camera = self.slide.properties.get('zeiss.camera')
        if zeiss_camera == 'Axiocam705c':
            self.dzslide.display_gamma = 0.45
        else:
            self.dzslide.display_gamma = None

        print('total %d dz levels' % len(self.dzslide.level_dimensions))
        # deepzoom
        self.level_scales = [1.0 / d for d in self.slide.level_downsamples]

        self.scale = 1.0
        self.position = 0
        # slide coordinates of point located at canvas (0,0)
        self.slide_x, self.slide_y = 0, 0

        self.dialog = False  # status of popup dialog
        #self.size_x = WIN_X
        #self.size_y = WIN_Y
        self.lastzoom_time = time.time()
        self.birdview, self.birdview_upleft = gen_birdview(self.slide)
        self.birdview_scale = min(float(BIRDVIEW_W) / self.slide.width,
                                  float(BIRDVIEW_H) / self.slide.height)

        self.press_inside_birdview = False
        self.crosshair = gen_crosshair()
        self.crosshair_upleft = ()
        self.is_edit_mode = False
        self.is_add_mode = False
        self.is_sam_mode = False
        self.is_del_mode = False
        self.cur_mask_id = None

        self.mask_to_hide = ()
        self.showmasklayer = True
        self.masklayer = None
        self.db_needcommit = False
        self.db_lastcommit = time.time()

        self.polygon_cache = {}
        self.polygon_pt_order_cache = {}
        self.polygon_vertex = 0  # pixel value of current vertex
        self.polygon_split_pair = [None, None]

        self.slide_buf_cache = {'img_resized': None, 'key': None}
        self.pixbuf = None

        self.reset_sam()
        self.disable_sam = False
        if self.slide.width > 1500 and self.slide.height > 1500:
            self.disable_sam = True
            self.image = None
        else:
            self.image = read_whole(self.slide)

        if self.predictor is not None and self.image is not None:
            print('Loading image to predictor ...')
            self.predictor.set_image(self.image)
            print('        image loaded')

        row = query_db(self.slide.db, 'select max(id) from Mask', one=True)
        self.max_mask_id = row[0] if row[0] is not None else 0
        print('max mask id = %d' % self.max_mask_id)

    def reset_sam(self):
        self.sam_points = []
        self.is_sam_mode = False
        self.sam_masks = None
        self.sam_statusbar = ''
        self.sam_mask_idx = 0
        self.sam_crop = {}

        if self.btn_box_sam is not None:
            self.btn_box_sam.hide()

    def reset_del(self):
        self.is_del_mode = False


    def init_gtk(self):
        #self.set_title('Label Whole Slide')
        #self.window.connect('delete_event', self.delete_event)
        self.connect('destroy', Gtk.main_quit)

        self.box = Gtk.Box(spacing=2)
        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.add_events(Gdk.EventMask.KEY_PRESS_MASK
                           | Gdk.EventMask.KEY_RELEASE_MASK)
        self.connect("key_release_event", self._key_release)

        self.vbox_left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.vbox_left.set_events(Gdk.EventMask.BUTTON_PRESS_MASK
                                | Gdk.EventMask.BUTTON_RELEASE_MASK)

        btns = {}
        for bn in ('open', 'new', 'save', 'delete', 'cancel'):
            btns[bn] = Gtk.Button(label=bn.capitalize())
            btns[bn].set_can_focus(False)  # disable keyboard on button

        btns['open'].set_margin_top(8)
        btns['cancel'].set_margin_bottom(8)
        for _, b in btns.items():
            b.set_margin_start(8)
            b.set_margin_end(8)

        btns['open'].connect('clicked', self.on_btn_open_clicked)
        btns['new'].connect('clicked', self.on_btn_new_clicked)
        btns['save'].connect('clicked', self.on_btn_save_clicked)
        btns['delete'].connect('clicked', self.on_btn_delete_clicked)
        btns['cancel'].connect('clicked', self.on_btn_cancel_clicked)

        btn_box1 = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        for bn in ('open', 'new', 'save', 'delete', 'cancel'):
            btn_box1.pack_start(btns[bn], False, False, 0)

        btn_box_sam = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        btn_sam_it = Gtk.Button(label='SAM it')
        btn_sam_it.set_can_focus(False)
        btn_sam_it.set_margin_start(8)
        btn_sam_it.set_margin_end(8)
        btn_sam_it.connect('clicked', self.on_sam_it_clicked)
        btn_box_sam.pack_start(btn_sam_it, False, False, 0)

        btn_prev = Gtk.Button(label='<')
        btn_next = Gtk.Button(label='>')
        btn_prev.set_can_focus(False)
        btn_next.set_can_focus(False)
        # set_size_request cannot set width to smaller value, perhaps due to
        # padding
        #btn_prev.set_size_request(10, -1)
        #btn_next.set_size_request(10, -1)
        arrow_box = Gtk.Box(spacing=0)

        btn_prev.connect('clicked', self.on_sam_prev_clicked)
        btn_next.connect('clicked', self.on_sam_next_clicked)
        arrow_box.pack_start(btn_prev, False, False, 0)
        arrow_box.pack_start(btn_next, False, False, 0)
        #btn_box_sam.pack_start(arrow_box, False, False, 0)

        win_w, win_h = self.get_size()
        self.win_w, self.win_h = self.get_size()

        self.canvas = Gtk.DrawingArea()
        self.canvas.set_size_request(win_w - CR_LEFT, win_h - CR_BOTTOM)

        # DrawingArea needs mask to capture mouse
        self.canvas.add_events(Gdk.EventMask.POINTER_MOTION_MASK
                | Gdk.EventMask.BUTTON_PRESS_MASK
                | Gdk.EventMask.BUTTON_RELEASE_MASK
                | Gdk.EventMask.SCROLL_MASK)
        self.canvas.connect('draw', self.expose_event)
        self.canvas.connect("motion-notify-event", self._mouse_move)
        self.canvas.connect("button_press_event", self._mouse_press)
        self.canvas.connect("button_release_event", self._mouse_release)
        self.canvas.connect("scroll_event", self._mouseScroll)

        self.statusbar = Gtk.Statusbar()
        # its context_id - not shown in the UI but needed to uniquely identify
        # the source of a message
        self.context_id = self.statusbar.get_context_id("example")
        self.statusbar.push(self.context_id, 'waiting ...')

        switch = Gtk.Switch()
        switch.set_can_focus(False)  # disable keyboard on switch
        switch.set_margin_start(10)
        switch.set_margin_end(10)
        switch.set_state(True)
        switch.connect("notify::active", self.on_switch_toggled)
        lbl = Gtk.Label(label='Mask')
        #self.vbox_left.pack_start(self.iconview, False, False, 0)
        self.vbox_left.pack_start(btn_box1, False, False, 0)
        self.vbox_left.pack_start(switch, False, False, 0)
        self.vbox_left.pack_start(lbl, False, False, 0)
        self.vbox_left.pack_start(btn_box_sam, False, False, 0)

        self.vbox.pack_start(self.canvas, True, True, 0)
        self.vbox.pack_start(self.statusbar, True, True, 0)

        #self.box.pack_start(self.iconview, False, False, 0)
        self.box.pack_start(self.vbox_left, False, False, 0)

        self.box.pack_start(self.vbox, False, False, 0)
        self.add(self.box)
        self.show_all()  # takes 0.1x s
        # hide SAM buttons unless in SAM mode
        self.btn_box_sam = btn_box_sam
        self.btn_box_sam.hide()

        self._mouseX = self._mouseY = self.press_x = self.press_y = 0
        self.current_folder = None

    def on_switch_toggled(self, switch, state):
        if switch.get_active():
            self.showmasklayer = True
            self.draw_slide_buf(refreshmask=False, togglemask=True)
        else:
            self.showmasklayer = False
            self.draw_slide_buf(refreshmask=False, togglemask=True)

        self.queue_draw()

    def on_open_file_clicked(self):
        print('open file clicked')
        dialog = Gtk.FileChooserDialog(title="Please choose a file",
                                       parent=self,
                                       action=Gtk.FileChooserAction.OPEN)
        if self.current_folder:
            dialog.set_current_folder(self.current_folder)

        dialog.add_buttons(Gtk.STOCK_CANCEL,
                           Gtk.ResponseType.CANCEL,
                           Gtk.STOCK_OPEN,
                           Gtk.ResponseType.OK)

        self.add_filters(dialog)
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            f = dialog.get_filename()
            self.current_folder = os.path.dirname(f)
            self.load_slide(f)

        dialog.destroy()

    def add_filters(self, dialog):
        filter_tif = Gtk.FileFilter()
        filter_tif.set_name("Tiff files")
        filter_tif.add_mime_type("image/tiff")
        dialog.add_filter(filter_tif)

        filter_any = Gtk.FileFilter()
        filter_any.set_name("Any files")
        filter_any.add_pattern("*")
        dialog.add_filter(filter_any)


    def on_btn_open_clicked(self, widget):
        self.on_open_file_clicked()

    def on_btn_new_clicked(self, widget):
        self.is_edit_mode = False
        self.is_add_mode = True
        self.polygon_cache['new'] = None
        self.cur_mask_id = 'new'

    def on_btn_save_clicked(self, widget):
        if self.is_add_mode:
            self.save_new_polygon()

        elif self.is_edit_mode:
            self.save_polygon_edit()

        elif self.is_sam_mode:
            self.save_sam_new()

    def on_btn_delete_clicked(self, widget):
        if not self.is_edit_mode:
            return

        self._delete_polygon(None, self.cur_mask_id)

    def on_btn_cancel_clicked(self, widget):
        if not (self.is_edit_mode or self.is_add_mode
             or self.is_sam_mode  or self.is_del_mode):

            return

        self.is_add_mode = False
        self.is_edit_mode = False
        self.cur_mask_id = 0
        self.mask_to_hide = ()
        self.polygon_cache['new'] = None
        self.polygon_split_pair = [None, None]
        self.reset_sam()
        self.reset_del()

        self.draw_slide_buf()  # do not draw on screen yet
        self.queue_draw()

    def save_polygon_edit(self):
        # mask occupied tiles before edit
        row = query_db(self.slide.db, 'select x,y,w,h from Mask WHERE id=?',
            (int(self.cur_mask_id),), one=True)
        x,y,w,h = row
        addr_before = get_mask_tile_address(self.dzslide, x, y, w, h)

        points = self.polygon_cache[self.cur_mask_id]
        x,y,w,h,maskarr = polygon2mask(points)
        # mask occupied tiles after edit
        addr_after = get_mask_tile_address(self.dzslide, x, y, w, h)

        sql = ('UPDATE Mask SET x=?,y=?,w=?,h=?,mask=?,polygon=?,is_edit=1  '
               'WHERE id=?')
        #blobm = narray2pngblob(maskarr)
        blobm = mask2rleblob(maskarr)

        # change polygon coordinates to the same as cropped mask
        blobp = narray2blob(points - np.array([y, x]))
        self.slide.db.execute(sql, (x,y,w,h,blobm,blobp,int(self.cur_mask_id)))

        args = addr_before + addr_after
        for arg in args:
            # arg = (dzlevel, c1, c2, r1, r2)
            self.slide.db.execute('delete from MaskTile where '
                'lid=? AND cid>=? AND cid<=? AND rid>=? AND rid<=? ', arg)

        self.slide.db.commit()
        print('Edited polygon saved')

        self.is_add_mode = False
        self.is_edit_mode = False
        self.cur_mask_id = 0
        self.mask_to_hide = ()
        self.polygon_cache['new'] = None

        self.draw_slide_buf()  # do not draw on screen yet
        self.queue_draw()

    def save_one_polygon(self, points):
        x,y,w,h,maskarr = polygon2mask(points)

        args = get_mask_tile_address(self.dzslide, x, y, w, h)
        self.max_mask_id += 1
        sql = ('INSERT INTO Mask (x,y,w,h,mask,polygon,is_edit,id) '
               ' VALUES(?,?,?,?,?,?,?,?)')
        #blobm = narray2pngblob(maskarr)
        blobm = mask2rleblob(maskarr)
        blobp = narray2blob(points - np.array([y, x]))

        self.slide.db.execute(sql, (x,y,w,h,blobm,blobp,1,self.max_mask_id))
        for arg in args:
            # arg = (dzlevel, c1, c2, r1, r2)
            self.slide.db.execute('delete from MaskTile where '
                'lid=? AND cid>=? AND cid<=? AND rid>=? AND rid<=? ', arg)

        self.slide.db.commit()
        print('New polygon saved')

    def save_new_polygon(self):
        points = self.polygon_cache['new']
        self.save_one_polygon(points)

        self.is_add_mode = False
        self.is_edit_mode = False
        self.cur_mask_id = 0
        self.mask_to_hide = ()
        self.polygon_cache['new'] = None

        self.draw_slide_buf()  # do not draw on screen yet
        self.queue_draw()

    def save_sam_new(self):
        if not self.sam_crop:
            return

        # manaul specify new mask id. The auto generated mask id may be
        # unique in db, but it may exists on canvas
        sql = ('INSERT INTO Mask (x,y,w,h,mask_area,is_edit,mask,id) '
               ' VALUES(?,?,?,?,?,?,?,?)')
        self.max_mask_id += 1
        x, y, w, h, area = (self.sam_crop['x'], self.sam_crop['y'],
                            self.sam_crop['w'], self.sam_crop['h'],
                            self.sam_crop['area'])
        self.slide.db.execute(sql, (x, y, w, h, area, 0,
                                    mask2rleblob(self.sam_crop['mask']),
                                    self.max_mask_id))
        # regenerate tiles
        args = get_mask_tile_address(self.dzslide, x, y, w, h)
        for arg in args:
            self.slide.db.execute('delete from MaskTile where '
                'lid=? AND cid>=? AND cid<=? AND rid>=? AND rid<=? ', arg)

        self.slide.db.commit()
        self.statusbar.push(self.context_id, 'New mask saved')

        self.reset_sam()
        self.draw_slide_buf()  # do not draw on screen yet
        self.queue_draw()

    def canvasxy_to_slidexy(self, x, y):
        """ given coordinates (x, y) on canvas, return its coordinates on
        level 0 slide as tuple """
        return (int(self.slide_x + x / self.scale),
                int(self.slide_y + y / self.scale))

    def slidexy_to_canvasxy(self, x, y):
        return ((x - self.slide_x) * scale, (y - self.slide_y) * scale)

    def slidetopleft_with_newscale(self, x, y, newscale):
        """ after zoom in/out at canvas (x, y), find the level 0 slide coordinates
        of canvas (0, 0) under new scale """
        #newlevel = self.find_level_from_scale(newscale)
        f = (newscale - self.scale ) / (self.scale * newscale)
        slide_x1 = self.slide_x + f * x
        slide_y1 = self.slide_y + f * y

        return (slide_x1, slide_y1)

    def find_level_from_scale(self, scale):
        """ find the best slide level for read_region
        read a big region is expensive, resize a big image is also expensive
        """

        if scale > 0.7:  # continue zoom in on level 0 image
            return 0

        dim = self.slide.level_count
        for n in range(dim):
            if  self.level_scales[n] > scale:
                level = n + 1
                break

        level = min(level, dim - 1)

        return level

    def expose_event(self, _unused, cr, data=None):
        win_w, win_h = self.get_size()
        if win_w != self.win_w or win_h != self.win_h:
            print('resize win_w, win_h', win_w, win_h)
            self.win_w, self.win_h = win_w, win_h
            #self.canvas.set_size_request(win_w-50, win_h-40)
            self.canvas.set_size_request(win_w - CR_LEFT, win_h - CR_BOTTOM)
            self._set_min_scale()
            # drag window will emit many resizing events, it is expensive to
            # update view on each size change. Manually pan the window to force
            # window redraw

        if not self.pixbuf:
            self.init_scale()
            self.draw_slide_buf()

        cr.set_source_surface(self.pixbuf, 0, 0)
        cr.paint()

        return

    def find_dz_level(self, scale):
        """ find the closest level with higher resolution
        Returns:
            (level, scale of that level) """
        nlevels = self.dzslide.level_count
        #dprint("slide has %d levels" % nlevels)
        if scale > 1:
            return (nlevels - 1, 1.0)

        level = nlevels - 1
        level_scale = 1.0
        while level > 0:
            if scale > level_scale:
                break

            level_scale *= 0.5
            level -= 1

        res = (level + 1, level_scale * 2)
        return res

    def read_dzregion(self, upleft_cord, dzlevel, dim, ismasktile=False):
        t0 = time.time()
        x, y = upleft_cord
        w, h = dim
        c1, r1 = get_tile_address(x, y)
        c2, r2 = get_tile_address(x+w-1, y+h-1)
        tw = th = TILE_SIZE
        crw, crh = self.get_canvas_size()
        cvs_w = (c2 - c1 + 1) * tw + TILE_OVERLAP * 2
        cvs_h = (r2 - r1 + 1) * th + TILE_OVERLAP * 2
        channels = 4 if ismasktile else 3
        cvs = np.zeros((cvs_h, cvs_w, channels), dtype=np.uint8)

        level_tile = self.dzslide.level_tiles[dzlevel]
        cmax, rmax = level_tile[0] - 1, level_tile[1] - 1

        x0 = 0
        for i in range(c2 - c1 + 1):
            c = c1 + i

            txoffset = 0 if c == 0 else TILE_OVERLAP
            tile_w = TILE_SIZE
            if c > 0:
                tile_w = TILE_SIZE + TILE_OVERLAP

            if c > cmax or c < 0:
                x0 += TILE_SIZE
                continue

            y0 = 0
            for j in range(r2 - r1 + 1):
                r = r1 + j
                tyoffset = 0 if r == 0 else TILE_OVERLAP

                if r > rmax or r < 0:
                    y0 += TILE_SIZE
                    continue

                t1 = time.time()
                if ismasktile:
                    _, tile, needcommit = get_or_set_masktile(self.slide.db,
                                               self.dzslide, dzlevel, (c, r))
                else:
                    tile, needcommit = get_or_set_tile(self.slide.db,
                                           self.dzslide, dzlevel, (c, r))

                self.db_needcommit = (self.db_needcommit or needcommit)

                h1, w1, _ = tile.shape
                x1 = x0 + w1 - txoffset
                y1 = y0 + h1 - tyoffset
                t = tile[tyoffset:, txoffset:]  # crop overlap
                cvs[y0: y1, x0:x1] = t
                y0 += TILE_SIZE

            x0 += TILE_SIZE

        # crop the tile canvas to fit screen
        tx, ty = dzxy_to_tilexy(x, y)
        outx = max(0, tx)
        outy = max(0, ty)
        img = Image.fromarray(cvs[outy:outy+h, outx:outx+w])

        out = img.convert('RGBA')
        return out


    def draw_slide_buf(self, refreshmask=True, togglemask=False):
        """ it needs to know the scale and topleft slide_x slide_y """

        # batch commit for inserting tiles
        if self.db_needcommit and (time.time() - self.db_lastcommit) > 5:
            self.slide.db.commit()
            self.db_needcommit = False
            self.db_lastcommit = time.time()

        t0 = time.time()
        sx1, sy1 = self.slide_x, self.slide_y
        crw, crh = self.get_canvas_size()

        buf_key = 'slidexy_%d_%d_scale_%.8f_crwh_%d_%d' % (sx1, sy1,
                self.scale, crw, crh)

        if ((self.is_edit_mode or self.is_add_mode or self.is_sam_mode)
            and buf_key == self.slide_buf_cache['key']
            and not refreshmask):

            if togglemask:
                img_resized = self.slide_buf_cache['img_resized_wo_mask'].copy()
                if self.showmasklayer:
                    img_resized.alpha_composite(self.masklayer)

                img_resized.alpha_composite(self.birdview, (crw - BIRDVIEW_W, 0))
                img_resized.alpha_composite(self.crosshair, self.crosshair_upleft)
                # replace the cache with toggled mask
                self.slide_buf_cache['img_resized'] = img_resized.copy()
            else:
                img_resized = self.slide_buf_cache['img_resized'].copy()

            if self.is_sam_mode:
                img_resized.alpha_composite(self.sam_layer)
            else:
                img_resized.alpha_composite(self.polygon_layer)

            self.pixbuf = surface_from_pil(img_resized)
            #print('draw_slide_buf cache mode use %.2f s' % (time.time() - t0))
            return

        # convert to coordinates on level n
        #print('self.scale = ' , self.scale)
        #sx1, sy1 = int(sx1), int(sy1)
        #sw1, sh1 = int(crw / self.scale), int(crh / self.scale)

        dzlevel, dzscale = self.find_dz_level(self.scale)
        dzx, dzy = int(self.slide_x * dzscale), int(self.slide_y * dzscale)
        dzw, dzh = int(crw / self.scale * dzscale), int(crh / self.scale * dzscale)

        if refreshmask:
            masklayer = self.read_dzregion((dzx, dzy), dzlevel, (dzw, dzh), True)
            if self.mask_to_hide:
                hx,hy,hw,hh,hmask = self.mask_to_hide
                tmp = np.array([hx-self.slide_x,hy-self.slide_y,hw,hh],
                        dtype=np.float64)
                tmp *= dzscale
                hx,hy,hw,hh = [int(t) for t in tmp]

                masklayer_arr = np.array(masklayer)
                himg = Image.fromarray(hmask)
                himg = himg.resize((hw, hh))
                hmask = np.array(himg)

                # the mask to hide may be partially outside masklayer
                hmaskdz = np.zeros((dzh, dzw), dtype=bool)
                _put_one_mask_in_region(hmaskdz, True, hx, hy, hmask, dzw, dzh)
                masklayer_arr[hmaskdz] = 0
                masklayer = Image.fromarray(masklayer_arr)

            #dprint('draw_slide_buf loc 0.2 uses %.2f s' % (time.time() - t0))
            if dzw != crw or dzh != crh:
                self.masklayer = masklayer.resize((crw, crh))
                #dprint('draw_slide_buf loc 0.3 uses %.2f s' % (time.time() - t0))
            else:
                self.masklayer = masklayer
                #dprint('draw_slide_buf loc 0.4 uses %.2f s' % (time.time() - t0))

        snapshot = self.read_dzregion((dzx, dzy), dzlevel, (dzw, dzh))

        #dprint('draw_slide_buf loc 0.5 uses %.2f s' % (time.time() - t0))
        if dzw != crw or dzh != crh:
            #print('    resize dz image from (dzw, dzh) to (crw, crh)', (dzw,
            #    dzh), (crw, crh))
            #tr = time.time()
            img_resized = snapshot.resize((crw, crh))
            #print('    resize dz image use %.4f s' % (time.time() - tr))
        else:
            img_resized = snapshot

        #print('draw_slide_buf loc 1 uses %.2f s' % (time.time() - t0))
        self.slide_buf_cache['img_resized_wo_mask'] = img_resized.copy()

        if self.showmasklayer:
            img_resized.alpha_composite(self.masklayer)

        img_resized.alpha_composite(self.birdview, (crw - BIRDVIEW_W, 0))
        img_resized.alpha_composite(self.crosshair, self.crosshair_upleft)

        self.slide_buf_cache['img_resized'] = img_resized.copy()
        self.slide_buf_cache['key'] = buf_key

        if self.is_edit_mode or self.is_add_mode:
            img_resized.alpha_composite(self.polygon_layer)
        elif self.is_sam_mode:
            img_resized.alpha_composite(self.sam_layer)

        self.pixbuf = surface_from_pil(img_resized)
        #print('draw_slide_buf uses %.2f s' % (time.time() - t0))

    def get_canvas_size(self):
        """ the canvas height is less than window height because the bottom of
        window is for statusbar """
        win_w, win_h = self.get_size()
        return (win_w - CR_LEFT, win_h - CR_BOTTOM)

    def _set_min_scale(self):
        """ find a scale and set to closest to 1/2^n, so that when scale up and
        down, then tile does not need to be resized """

        self.win_w, self.win_h = self.get_size()
        ratio_w = float(self.win_w) / self.slide.width
        ratio_h = float(self.win_h) / self.slide.height
        m = min(ratio_w, ratio_h)
        self.min_scale = 1.0
        n = 1
        while self.min_scale > m:
            self.min_scale = 1.0 / 2 ** n
            n += 1

        dprint('set self.min_scale to %.6f' % self.min_scale)

    def _set_dzslide_tile_min_level(self):
        """ set min_level of dzslide tile, use it to delete tiles on all levels
        at a location """

        q = query_db(self.slide.db, 'select min(lid) from SlideTile', one=True)
        self.dzslide.minlevel = q[0] if q[0] is not None else 100
        dprint('set dzslide.minlevel to %d' % self.dzslide.minlevel)

    def init_scale(self):
        """ start the window with the last slide level """
        crw, crh = self.get_canvas_size()
        w, h = self.slide.width, self.slide.height
        self.level = self.slide.level_count - 1


        """
        if self.slide.mpp < 0.25:
            self.initscale = 0.125
        elif self.slide.mpp < 0.5:
            self.initscale = 0.125
        else:
            self.initscale = START_SCALE
        """

        self._set_min_scale()
        self.initscale = self.min_scale
        dprint('initial scale = %.4f' % self.initscale)
        self.scale = self.initscale
        self.update_crosshair_upleft()

        return

    def _mouseScroll(self, cr, event):

        if self.is_inside_birdview(event.x, event.y):
            return

        now = time.time()
        if (now - self.lastzoom_time) < DELAY_UPDATE:
            return

        self._mouseX, self._mouseY = event.x, event.y

        if self.scale > 1.0:
            scale_step = 0.5
            scale_mul = 0
        else:
            scale_step = 0
            scale_mul = 2

        if (event.direction == Gdk.ScrollDirection.DOWN):
            scale_step *= -1.0
            scale_mul = 0.5 if scale_mul else 0

        if scale_mul:
            newscale = self.scale * scale_mul

        if scale_step:
            newscale = self.scale + scale_step

        if newscale >= MAX_SCALE:
            newscale = MAX_SCALE

        if newscale <= self.min_scale:
            newscale = self.min_scale

        newlevel = self.find_level_from_scale(newscale)

        # find the new topleft points on level 0 slide after zoom, so that
        # point under mouse remain at same canvas location ( unless topleft
        # point is outside of slide, then the point under mouse may move )
        sx1, sy1 = self.slidetopleft_with_newscale(
                self._mouseX, self._mouseY, newscale)

        # sx1 and sy1 may outside of slide
        self.slide_x, self.slide_y = self._normalize_slidexy(sx1, sy1)
        self.level = newlevel
        self.scale = newscale

        self.update_crosshair_upleft()

        self.update_statusbar()
        self.update_polygon_layer(self.cur_mask_id)
        if self.sam_masks is not None or self.sam_points:
            self.update_sam_layer()
        self.draw_slide_buf()  # do not draw on screen yet
        self.queue_draw()
        self.lastzoom_time = time.time()
        return

    def show_edit_menu(self, mask_id):
        m1 = Gtk.MenuItem(label='Edit polygon')
        m2 = Gtk.MenuItem(label='Delete polygon')
        m3 = Gtk.MenuItem(label='Mark as bad')
        m4 = Gtk.MenuItem(label='Enter Delete Mode')
        m1.connect('activate', self._edit_polygon, mask_id)
        m2.connect('activate', self._delete_polygon, mask_id)
        m3.connect('activate', self._mark_bad, mask_id)
        m4.connect('activate', self._enter_delete_mode)
        menu = Gtk.Menu()
        menu.append(m1)
        menu.append(m2)
        menu.append(m3)
        menu.append(m4)
        menu.show_all()
        menu.popup(None, None, None, None, 0, Gtk.get_current_event_time())

    def show_add_menu(self):
        m1 = Gtk.MenuItem(label='Add polygon')
        m1.connect('activate', self._add_polygon)
        m2 = Gtk.MenuItem(label='Add SAM points')
        m2.connect('activate', self._add_sam)

        menu = Gtk.Menu()
        menu.append(m2)
        menu.append(m1)
        menu.show_all()
        menu.popup(None, None, None, None, 0, Gtk.get_current_event_time())

    def show_add_delete_point_menu(self, point_idx):
        m1 = Gtk.MenuItem(label='Add point righthand')
        m2 = Gtk.MenuItem(label='Add point lefthand')
        m12 = Gtk.MenuItem(label='Add point RL')
        m3 = Gtk.MenuItem(label='Delete point')
        m4 = Gtk.MenuItem(label='Split Polygon')

        m1.connect('activate', self._add_point, point_idx, 1)
        m2.connect('activate', self._add_point, point_idx, 0)
        m12.connect('activate', self._add_lr_points, point_idx)
        m3.connect('activate', self._delete_point, point_idx)
        m4.connect('activate', self._split_polygon, point_idx)
        menu = Gtk.Menu()
        menu.append(m12)
        menu.append(m1)
        menu.append(m2)
        menu.append(m3)
        menu.append(m4)
        menu.show_all()
        menu.popup(None, None, None, None, 0, Gtk.get_current_event_time())

    def get_set_pt_order(self, mask_id):
        if mask_id in self.polygon_pt_order_cache:
            pt_order = self.polygon_pt_order_cache[mask_id]
        else:
            points = self.polygon_cache[mask_id]
            pt_order = get_polygon_direction(points)
            self.polygon_pt_order_cache[mask_id] = pt_order
        return pt_order

    def _add_point(self, widget, idx, righthand=1):
        points = self.polygon_cache[self.cur_mask_id]

        pt_order = self.get_set_pt_order(self.cur_mask_id)

        last = len(points) - 1
        if ((pt_order == RIGHT_HAND and righthand)
            or (pt_order == LEFT_HAND and righthand == 0)):

            if idx == last:
                m = (points[0] + points[-1]) / 2
                out = np.append(points, [m], axis=0)
            else:
                out = points[:idx+1]
                m = (points[idx] + points[idx + 1]) / 2
                out = np.append(out, [m], axis=0)
                out = np.append(out, points[idx+1:], axis=0)

        else:
            if idx == 0:
                m = (points[0] + points[-1]) / 2
                out = np.array([m])
                out = np.append(out, points, axis=0)
            else:
                out = points[:idx]
                m = (points[idx - 1] + points[idx]) / 2
                out = np.append(out, [m], axis=0)
                out = np.append(out, points[idx:], axis=0)

        self.polygon_cache[self.cur_mask_id] = out

        self.update_polygon_layer(self.cur_mask_id)
        self.draw_slide_buf(refreshmask=False)
        self.queue_draw()

    def _add_lr_points(self, widget, idx):
        """ add two points to both side"""
        self._add_point(widget, idx, 1)
        self._add_point(widget, idx, 0)


    def _split_polygon(self, widget, idx):
        # this is the first time select split polygon
        if not self.polygon_split_pair[0]:
            self.polygon_split_pair[0] = idx
            msg = 'Right click on next point\nthen select split polygon'
            self.show_popup_dialog(msg)
            return

        tmp = [self.polygon_split_pair[0], idx]
        tmp.sort()
        idx0, idx1 = tmp
        points = self.polygon_cache[self.cur_mask_id]
        points = points.copy()
        if idx0 == 0:
            plg0 = points[:idx1+1]
            plg1 = points[idx1:]
        else:
            plg0 = points[idx1:]
            plg0 = np.append(plg0, points[:idx0+1], axis=0)
            plg1 = points[idx0: idx1+1]

        # save one new splitted polygon and update the current polygon
        self.polygon_cache[self.cur_mask_id] = plg0
        self.polygon_pt_order_cache.pop(self.cur_mask_id, None)
        self.save_one_polygon(plg1)
        self.save_polygon_edit()
        self.polygon_split_pair = [None, None]

    def show_popup_dialog(self, msg=''):
        dialog = Gtk.MessageDialog(transient_for=self, flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text='Usage')
        dialog.format_secondary_text(msg)

        dialog.run()
        dialog.destroy()

    def _delete_point(self, widget, point_idx):
        points = self.polygon_cache[self.cur_mask_id]
        self.polygon_cache[self.cur_mask_id] = np.delete(
                points, [point_idx,], axis=0)

        self.update_polygon_layer(self.cur_mask_id)
        self.draw_slide_buf(refreshmask=False)
        self.queue_draw()

    def _add_polygon(self, widget):
        self.is_edit_mode = False
        self.is_sam_mode = False
        self.is_add_mode = True
        self.polygon_cache['new'] = None
        self.cur_mask_id = 'new'

    def _add_sam(self, widget):
        self.is_edit_mode = False
        self.is_sam_mode = True
        self.is_add_mode = False
        self.btn_box_sam.show()
        # TODO show sam seg button

    def on_sam_prev_clicked(self, widget):
        if self.sam_masks is None:
            return

        self.sam_mask_idx -= 1
        self.sam_change_mask()

    def on_sam_next_clicked(self, widget):
        if self.sam_masks is None:
            return

        self.sam_mask_idx += 1
        self.sam_change_mask()

    def sam_change_mask(self):
        n, h, w = self.sam_masks.shape
        if self.sam_mask_idx < 0:
            self.sam_mask_idx = n - 1
        elif self.sam_mask_idx == n:
            self.sam_mask_idx = 0

        self.sam_statusbar = (
'Showing %d of %d predicted masks, use Left and Right key to change mask,'
'Enter to save mask' % (
                self.sam_mask_idx + 1, len(self.sam_masks))
        )

        self.update_sam_layer()
        self.draw_slide_buf(refreshmask=False)
        self.queue_draw()

    def on_sam_it_clicked(self, widget):
        # self.sam_points is of [(y, x), ...], switch it to (x, y)
        input_points = np.array([(p[1], p[0]) for p in self.sam_points])
        input_labels = np.array([1] * len(self.sam_points))
        pickle_outfn = 'delme.pickle'
        print('generating masks...')
        if not reuse_pickle:
            res = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )

            with open(pickle_outfn, 'wb') as f:
                pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
                print('predicte dumped to %s' % pickle_outfn)
            """
            """
        else:
            with open(pickle_outfn, 'rb') as f:
                res = pickle.load(f)
                print('masks loaded to %s' % pickle_outfn)

        masks, scores, logits = res
        self.sam_mask_idx = 0
        self.sam_statusbar = 'showing %d of %d predicted masks' % (
                self.sam_mask_idx + 1, len(masks))

        self.sam_masks = masks
        self.update_sam_layer()
        self.draw_slide_buf(refreshmask=False)
        self.queue_draw()

    def polygon_by_mask_id(self, mask_id):
        """ find edge points for a mask_id, return a float64 narray.
        use level 0 slide coordinates """

        # polygon corrdinates is [(y, x), ...], NOT (x, y)
        if mask_id in self.polygon_cache:
            return self.polygon_cache[mask_id]

        row = query_db(self.slide.db,
            'select x,y,w,h,mask,polygon from Mask WHERE id=?',
            (int(mask_id),), one=True)
        x,y,w,h,blobm, blobp = row
        if blobp:
            s = blob2narray(blobp)
            s = np.float64(s)
            s += np.array([y, x])
            self.polygon_cache[mask_id] = s
            return s

        mask = rleblob2mask(blobm)

        # leave 10 pixels at each side of mask
        cvs = np.zeros((h+20, w+20), dtype=np.uint8)
        cvs[10:h+10, 10:w+10][mask] = 100
        s = mask2shape(cvs, self.slide.mpp)
        # to coordinates on level 0 slide
        s += np.array([y-10, x-10])
        s = np.float64(s)
        self.polygon_cache[mask_id] = s
        return s

    def update_polygon_layer(self, mask_id):
        if not mask_id:
            return

        # polygon is cached
        s = self.polygon_by_mask_id(mask_id)
        s = s.copy()
        s -= np.array([self.slide_y, self.slide_x])
        s *= self.scale  # to coordinates on GTK canvas
        s = np.int32(s)

        crw, crh = self.get_canvas_size()

        self.polygon = draw_one_polygon(crw, crh, s)
        im = np.zeros((crh, crw, 4), dtype=np.uint8)
        im[:,:,1][self.polygon != 0] = 255
        im[:,:,3][self.polygon != 0] = 128
        self.polygon_layer = Image.fromarray(im)

    def update_sam_layer(self):
        """ paint SAM input points, or the predicted mask """
        crw, crh = self.get_canvas_size()
        cvm = np.zeros((crh, crw), dtype=np.uint8)
        colors = (CYAN, BLUE, YELLOW, TOMATO, RED)

        if self.sam_masks is None:
            # convert x,y on slide to x,y on canvas
            cvs_sam_points = np.array(self.sam_points, dtype=np.float64)
            cvs_sam_points -= np.array([self.slide_y, self.slide_x])
            cvs_sam_points *= self.scale
            for p in cvs_sam_points:
                rr, cc = ski_draw.disk(p, 6)
                cvm[rr, cc] = 1
            color = BLUE
            mask_alpha = 255
        else:
            crop = mask2crop(self.sam_masks[self.sam_mask_idx])
            w = int(float(crop['w']) * self.scale)
            h = int(float(crop['h']) * self.scale)
            mx = int(float(crop['x'] - self.slide_x) * self.scale)
            my = int(float(crop['y'] - self.slide_y) * self.scale)
            dprint('x mx %d %d, y my %d %d' % (
                crop['x'], mx, crop['y'], my))
            mask = utimage.resize_mask(crop['mask'], w, h)
            _put_one_mask_in_region(cvm, 1, mx, my, mask, crw, crh)
            self.sam_crop = crop
            color = colors[self.sam_mask_idx]
            mask_alpha = 128

        im = np.zeros((crh, crw, 4), dtype=np.uint8)
        im[:,:,0][cvm != 0] = color[0]
        im[:,:,1][cvm != 0] = color[1]
        im[:,:,2][cvm != 0] = color[2]
        im[:,:,3][cvm != 0] = mask_alpha

        self.sam_layer = Image.fromarray(im)
        if self.sam_statusbar:
            self.statusbar.push(self.context_id, self.sam_statusbar)

    def _edit_polygon(self, widget, mask_id):
        print('debug: edit polygon of mask', mask_id)
        self.cur_mask_id = int(mask_id)
        row = query_db(self.slide.db, 'select x,y,w,h,mask from Mask WHERE id=?',
            (int(mask_id),), one=True)
        x,y,w,h,blobm = row
        mask = rleblob2mask(blobm)
        # TODO add mask to cache, use it to hide mask when edit
        self.mask_to_hide = (x,y,w,h,mask)

        self.update_polygon_layer(mask_id)
        self.is_edit_mode = True
        self.draw_slide_buf()  # do not draw on screen yet
        self.queue_draw()

    def _delete_polygon(self, widget, mask_id):
        # prepare remove tiles contain this mask on all dzlevels
        row = query_db(self.slide.db, 'select x,y,w,h from Mask WHERE id=?',
            (int(mask_id),), one=True)
        x,y,w,h = row
        args = get_mask_tile_address(self.dzslide, x, y, w, h)

        for arg in args:
            # arg = (dzlevel, c1, c2, r1, r2)
            self.slide.db.execute('delete from MaskTile where '
                'lid=? AND cid>=? AND cid<=? AND rid>=? AND rid<=? ', arg)

        self.slide.db.execute('delete from Mask where id=?', (int(mask_id),))
        self.slide.db.commit()
        self.is_edit_mode = False
        self.mask_to_hide = ()
        print('mask %d deleted' % mask_id)
        self.draw_slide_buf()  # do not draw on screen yet
        self.queue_draw()

    def _mark_bad(self, widget, mask_id):
        # prepare remove tiles contain this mask on all dzlevels
        row = query_db(self.slide.db, 'select x,y,w,h from Mask WHERE id=?',
            (int(mask_id),), one=True)
        x,y,w,h = row
        args = get_mask_tile_address(self.dzslide, x, y, w, h)

        for arg in args:
            # arg = (dzlevel, c1, c2, r1, r2)
            self.slide.db.execute('delete from MaskTile where '
                'lid=? AND cid>=? AND cid<=? AND rid>=? AND rid<=? ', arg)

        self.slide.db.execute('update Mask set is_bad=2 where id=?',
                              (int(mask_id),))
        self.slide.db.commit()
        self.is_edit_mode = False
        self.mask_to_hide = ()
        dprint('mark mask %d as bad' % mask_id)
        self.draw_slide_buf()  # do not draw on screen yet
        self.queue_draw()

    def _enter_delete_mode(self, widget):
        self.is_edit_mode = False
        self.is_add_mode = False
        self.is_sam_mode = False
        self.is_del_mode = True

        self.statusbar.push(self.context_id,
            'Delete mode, press x to delete mask under mouse cursor')

    def _mouse_press(self, widget, event):
        if event.button == 3:  # right click
            if self.is_inside_birdview(event.x, event.y):
                return

            # project canvas location to level 0 slide and get the mask id
            sx, sy = self.canvasxy_to_slidexy(event.x, event.y)
            col, row =  get_tile_address(sx, sy)  # on top level deepzoom
            toplevel = self.dzslide.level_count - 1
            masks, _, _ = get_or_set_masktile(self.slide.db,
                    self.dzslide, toplevel, (col, row), needmask=True)
            tx, ty = dzxy_to_tilexy(sx, sy)
            mid = masks[ty, tx]

            if not self.is_edit_mode:
                if  mid == 0:
                    self.show_add_menu()
                else:
                    self.show_edit_menu(mid)

            # handle add/delete vertex to polygon
            if self.is_edit_mode:
                self.polygon_vertex = self.get_polygon_vertex(event.x, event.y)
                idx = self.polygon_vertex - 1
                if self.polygon_vertex != 0:
                    self.show_add_delete_point_menu(idx)
                else:
                    return

            return

        self.press_x, self.press_y = event.x, event.y
        if self.is_inside_birdview(event.x, event.y):
            self.press_inside_birdview = True
        else:
            self.press_inside_birdview = False

        if self.is_edit_mode:
            self.polygon_vertex = self.get_polygon_vertex(event.x, event.y)

    def get_polygon_vertex(self, x, y):
        sx, sy = self.canvasxy_to_slidexy(x, y)
        sx -= self.slide_x
        sy -= self.slide_y
        sx *= self.scale
        sy *= self.scale
        pidxplusone = self.polygon[int(sy), int(sx)]
        #print('get_polygon_vertex pidxplusone = ', pidxplusone)
        return pidxplusone

    def _normalize_slidexy(self, sx, sy):
        """ normalize slide_x and slide_y so that they are inside slide """
        sx, sy = int(sx), int(sy)
        w, h = self.slide.width, self.slide.height
        sx = 0 if sx < 0 else sx
        sx = w - 1 if sx >= w else sx
        sy = 0 if sy < 0 else sy
        sy = h - 1 if sy >= h else sy
        return (sx, sy)

    def _key_release(self, widget, event):
        if not (self.is_sam_mode or self.is_del_mode or self.is_edit_mode):
            return


        k = event.keyval
        if self.is_sam_mode and k == Gdk.KEY_Left:
            self.on_sam_prev_clicked(None)

        elif self.is_sam_mode and k == Gdk.KEY_Right:
            self.on_sam_next_clicked(None)

        elif self.is_sam_mode and (k == Gdk.KEY_Return or
                                   k == Gdk.KEY_KP_Return):
            # In add points to SAM mode, press Enter will run predict. Whereas
            # in review SAM mask mode, press Enter will save mask
            if self.sam_masks is None:
                self.on_sam_it_clicked(None)
            else:
                self.save_sam_new()

        elif k == Gdk.KEY_x and (self.is_del_mode or self.is_edit_mode):
            if self.is_inside_birdview(self._mouseX, self._mouseY):
                return

            # project canvas location to level 0 slide and get the mask id
            if self.is_del_mode:
                sx, sy = self.canvasxy_to_slidexy(self._mouseX, self._mouseY)
                col, row =  get_tile_address(sx, sy)  # on top level deepzoom
                toplevel = self.dzslide.level_count - 1
                masks, _, _ = get_or_set_masktile(self.slide.db,
                        self.dzslide, toplevel, (col, row), needmask=True)
                tx, ty = dzxy_to_tilexy(sx, sy)
                mid = masks[ty, tx]

                if mid != 0:
                    self._delete_polygon(widget, mid)

            elif self.is_edit_mode:  # delete one point of polygon
                self.polygon_vertex = self.get_polygon_vertex(self._mouseX,
                                                              self._mouseY)
                idx = self.polygon_vertex - 1
                if self.polygon_vertex != 0:
                    self._delete_point(widget, idx)

        elif k == Gdk.KEY_Escape:
            if self.is_sam_mode:
                self.reset_sam()
            elif self.is_del_mode:
                self.reset_del()

            self.draw_slide_buf(refreshmask=False)
            self.queue_draw()

    def _mouse_release(self, widget, event):
        if event.button == 3:  # right click
            return

        crw, crh = self.get_canvas_size()
        if self.press_inside_birdview:
            bv_x = self.press_x - (crw - BIRDVIEW_W)
            bv_y = self.press_y
            sx, sy = self.birdviewxy_to_slidexy(bv_x, bv_y)
            sx -= (crw / 2) / self.scale
            sy -= (crh / 2) / self.scale
            self.slide_x, self.slide_y = self._normalize_slidexy(sx, sy)

        elif self.is_edit_mode and self.polygon_vertex != 0:
            points = self.polygon_by_mask_id(self.cur_mask_id)
            idx = self.polygon_vertex - 1
            sx, sy = self.canvasxy_to_slidexy(event.x, event.y)
            self.polygon_cache[self.cur_mask_id][idx] = [sy, sx]

            # don't forget to reset vertex value to bg
            self.polygon_vertex = 0

            self.update_polygon_layer(self.cur_mask_id)
            self.draw_slide_buf(refreshmask=False)
            self.queue_draw()
            return

        elif self.is_add_mode:
            # add a new point to polygon if pointer hasn't move
            if (abs(event.x - self.press_x) < 2
                    and abs(event.y - self.press_y) < 5):
                sx, sy = self.canvasxy_to_slidexy(event.x, event.y)

                if self.polygon_cache['new'] is None:
                    self.polygon_cache['new'] = np.array([[sy, sx]],
                            dtype=np.float64)
                else:
                    # if come back to the first point means polygon is done
                    pidxplusone = self.polygon[int(event.y), int(event.x)]
                    if pidxplusone == 1:
                        p = self.polygon_cache['new'][0]
                    else:
                        p = [sy, sx]

                    self.polygon_cache['new'] = np.append(
                            self.polygon_cache['new'], [p], axis=0)

                    if pidxplusone == 1:
                        self.save_new_polygon()
                        return

                self.update_polygon_layer(self.cur_mask_id)
                self.draw_slide_buf(refreshmask=False)
                self.queue_draw()
                return

            # also pan
            self.slide_x -= int((event.x - self.press_x) / self.scale)
            self.slide_y -= int((event.y - self.press_y) / self.scale)

        elif self.is_sam_mode and self.sam_masks is None:
            # add a new point if pointer hasn't move
            if (abs(event.x - self.press_x) < 2
                    and abs(event.y - self.press_y) < 5):
                sx, sy = self.canvasxy_to_slidexy(event.x, event.y)
                self.sam_points.append((sy, sx))
                self.update_sam_layer()
                self.draw_slide_buf(refreshmask=False)
                self.queue_draw()
                return

            # also pan
            #self.slide_x -= int((event.x - self.press_x) / self.scale)
            #self.slide_y -= int((event.y - self.press_y) / self.scale)
            sx = self.slide_x - ((event.x - self.press_x) / self.scale)
            sy = self.slide_y - ((event.y - self.press_y) / self.scale)
            self.slide_x, self.slide_y = self._normalize_slidexy(sx, sy)

        else:
            sx = self.slide_x - ((event.x - self.press_x) / self.scale)
            sy = self.slide_y - ((event.y - self.press_y) / self.scale)
            self.slide_x, self.slide_y = self._normalize_slidexy(sx, sy)
            #print('new slide_x and slide_y', self.slide_x, self.slide_y)

        self.update_crosshair_upleft()
        self.update_polygon_layer(self.cur_mask_id)
        if self.sam_masks is not None or self.sam_points:
            self.update_sam_layer()
        self.draw_slide_buf()  # do not draw on screen yet
        self.queue_draw()

    def _mouse_move(self, widget, event):
        if self.is_sam_mode:
            return

        self._mouseX, self._mouseY = event.x, event.y
        if not self.is_del_mode:
            self.update_statusbar()

    def is_inside_birdview(self, x, y):
        crw, crh = self.get_canvas_size()
        if x > (crw - BIRDVIEW_W) and y < BIRDVIEW_H:
            return True

        return False

    def birdviewxy_to_slidexy(self, x, y):
        x -= self.birdview_upleft[0]
        y -= self.birdview_upleft[1]
        return (x / self.birdview_scale, y / self.birdview_scale)

    def update_statusbar(self):
        xonslide, yonslide = self.canvasxy_to_slidexy(self._mouseX, self._mouseY)
        self.statusbar.push(self.context_id,
            'Screen (%d, %d) Scale %.2f | Slide (%d, %d) (%.3f, %.3f)' % (
             self._mouseX, self._mouseY, self.scale,
             xonslide, yonslide,
             float(xonslide)/ self.slide.width,
             float(yonslide)/ self.slide.height))

        return

    def update_crosshair_upleft(self):
        """ birdview crosshair location as upper left corner on canvas """
        crw, crh = self.get_canvas_size()
        sx, sy = self.canvasxy_to_slidexy(crw / 2, crh / 2)
        cvs_center_on_bv_x = sx * self.birdview_scale
        cvs_center_on_bv_y = sy * self.birdview_scale
        x = crw - BIRDVIEW_W + self.birdview_upleft[0] + cvs_center_on_bv_x - 10
        y = self.birdview_upleft[1] + cvs_center_on_bv_y - 10
        if y < 0:
            y = 0
        if y > (BIRDVIEW_H - 20):
            y = BIRDVIEW_H - 20
        if x < (crw - BIRDVIEW_W):
            x = crw - BIRDVIEW_W
        if x > (crw - 20):
            x = crw -20

        self.crosshair_upleft = (int(x), int(y))

    def run(self):
        Gtk.main()


def main():
    SVSF = '/zroot/nfs/wei/aperio/Villin_and_Count_Glomeruli/2020-10-05_gt/Normctrl_C305.svs'
    SVSF = '/zroot/nfs/wei/aperio/480_PASH_small.svs'
    SVSF = os.path.join(APPDIR, '480_PASH_small.svs')
    if len(sys.argv) != 2:
        print('Usage: labelwsi.py slide_file')
        exit(1)

    slide_path = sys.argv[1]
    nav = Labelwsi(slide_path)
    nav.run()


if __name__ == "__main__":
    main()
