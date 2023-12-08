
import colorsys
import json
import math
import os
import re
import sqlite3
import time
import zlib
from io import BytesIO
from xml.etree.ElementTree import ElementTree, Element, SubElement

import numpy as np
import openslide
from skimage.transform import resize as ski_resize
from PIL import Image
from utmask import masklocation, mask2shape, mask2crop
from utmask import mask2rleblob, rleblob2mask

APPDIR = os.path.abspath(os.path.dirname(__file__))


# sliding window width 1333 * 0.5 um, height 800 * 0.5 um
STEP_X = 1333
STEP_Y = 800
REGION_OVERLAP = 200
MASK_OVERLAP = 10
APERIO_CS2_MPP = 0.5021
APERIO_CS2_MPP = 0.5000

MASK_FMT_PNG = 0
MASK_FMT_RLE = 1

# for casts detection
"""
STEP_X = 600
STEP_Y = 600
REGION_OVERLAP = 100
MASK_OVERLAP = 10  # not tile overlap
"""

TH = 200
MIN_COMMIT = 200

TILESIZE = 1022  # tilesize not include the overlapping pixel
TILEOVERLAP = 1
TILEOVERLAP2 =  TILEOVERLAP * 2


class UTSlide(openslide.OpenSlide):
    def __init__(self, slide_file, limit_bounds=True):
        """ slide is a openslide.OpenSlide object """
        #self.slide = openslide.OpenSlide(slide_file)
        if not os.path.exists(slide_file):
            print('%s does not exists' % slide_file)
            exit(1)

        super().__init__(slide_file)
        self.x = 0
        self.y = 0

        # sometimes shift next2 window for few pixels can improve mask
        # prediction. Use x_offset to adjust the shift width
        self.x_offset = 0
        self.y_offset = 0

        self.rows = -1
        self.cols = -1
        self.size = -1
        self.cord = [0, 0]  # current coordinate as [col, row]
        self.grid = np.array([])
        self.idx = 0
        self.commit_count = 0
        self.img = None
        self.dbfile = None
        self.db = None
        self.cur = None
        self.rndcolors = None  # array of random colors for glms

        self.width = int(self.properties.get(
                         openslide.PROPERTY_NAME_BOUNDS_WIDTH, 0))
        self.height = int(self.properties.get(
                          openslide.PROPERTY_NAME_BOUNDS_HEIGHT, 0))
        # aperio 20x 0.5, versa 0.2744
        default_mpp = APERIO_CS2_MPP  # mpp on aperio 20x 0.5
        self.mpp = float(self.properties.get(
                          openslide.PROPERTY_NAME_MPP_X, default_mpp))
        print('default_mpp = %4f, slide mpp = %.4f' % (default_mpp, self.mpp))

        self.step_x = int(STEP_X * default_mpp / self.mpp)
        self.step_y = int(STEP_Y * default_mpp / self.mpp)
        self.region_overlap = int(REGION_OVERLAP * default_mpp / self.mpp)
        self.mask_overlap = int(MASK_OVERLAP * default_mpp / self.mpp)
        if self.width == 0 or self.height == 0:
            self.width, self.height = self.dimensions

        self._l0_offset = (0, 0)
        if limit_bounds:
            self._l0_offset = tuple(int(self.properties.get(prop, 0))
                    for prop in (openslide.PROPERTY_NAME_BOUNDS_X,
                                 openslide.PROPERTY_NAME_BOUNDS_Y)
            )

        re_region = re.compile(r'openslide.region\[(\d+)\]\.(\w+)')
        regions = {}
        for k in self.properties:
            if not k.startswith('openslide.region['):
                continue
            m = re_region.match(k)
            if not m:
                continue

            r, t = m.group(1), m.group(2)
            if r not in regions:
                regions[r] = {}
            regions[r][t] = int(self.properties[k])

        #print(n, t, self.properties[k])
        self.regions = []
        for r,v in regions.items():
            self.regions.append({'x1': v['x'],
                                 'y1': v['y'],
                                 'x2': v['x'] + v['width'],
                                 'y2': v['y'] + v['height']})

        print('debug: width, height, mpp = ', self.width, self.height, self.mpp)
        print('       read_region step_x, step_y (%d, %d)' % (self.step_x,
                                                              self.step_y))
        self.init_size2()
        self.load_or_create_db(slide_file)
        self.get_maskfmt()
        self.convert_pngmask_to_rle()

    def load_or_create_db(self, slide_file):
        ''' attach a sqlite db to slide, use it to store mask'''
        dbdir = os.path.dirname(slide_file)
        bname = os.path.basename(slide_file)
        idx = bname.rfind('.')
        bname = bname[:idx]
        dbfile = os.path.join(dbdir, '%s.db' % bname)
        self.dbfile = dbfile

        if os.path.exists(dbfile):
            db = sqlite3.connect(dbfile, timeout=5)
            update_db_schema(db)
            db.close()
        else:
            print('creating database file %s' % dbfile)

        self.db = sqlite3.connect(dbfile, timeout=5)
        #self.db.execute('pragma journal_mode=wal;')
        self.db.row_factory = sqlite3.Row
        self.cur = self.db.cursor()
        fp = open(os.path.join(APPDIR, 'schema.sql'))
        sql = fp.read()
        self.cur.executescript(sql)
        self.db.commit()

    def get_maskfmt(self):
        r = query_db(self.db, 'select maskfmt from Info', one=True)
        self.maskfmt = r[0] if r else MASK_FMT_PNG

    def convert_pngmask_to_rle(self):
        if self.maskfmt == MASK_FMT_RLE:
            return

        print('convert PNG mask to RLE...')
        rows = query_db(self.db, 'select id,mask from Mask')
        sql_upd = 'update Mask set mask=? where id=?'
        cur = self.db.cursor()
        for r in rows:
            mid, mask = r[0], blob2mask(r[1])
            blob = mask2rleblob(mask)
            cur.execute(sql_upd, (blob, mid))

            self.commit_count += 1
            if self.commit_count % MIN_COMMIT == 0:
                self.db.commit()

        print('    converted %d PNG mask to RLE' % self.commit_count)

        r = query_db(self.db, 'select id from Info', one=True)
        if r:
            sql = 'UPDATE Info set maskfmt=? where id=?'
            self.db.execute(sql, (MASK_FMT_RLE, r[0]))
        else:
            sql = 'INSERT INTO Info (title,body,maskfmt) VALUES(?,?,?)'
            self.db.execute(sql, ('slideinfo', json.dumps({}), MASK_FMT_RLE))

        self.db.commit()
        self.commit_count = 0
        self.maskfmt = MASK_FMT_RLE

    def init_size(self):
        w, h = (self.width, self.height)
        rows, cols = math.ceil(h/STEP_Y), math.ceil(w/self.step_x)
        self.rows, self.cols = rows, cols
        self.size = rows * cols

    def init_emptiness_grid(self):
        """ find area with tissue by looking at green channel, using the
        level 1 image in SVS """
        # the level 1 image
        snapshot = self.read_region((0,0), 1, self.level_dimensions[1])
        print('debug self.level_dimensions[1] :', self.level_dimensions[1])
        ratio = int(self.level_dimensions[0][0] / self.level_dimensions[1][0])
        im = np.asarray(snapshot)  # m.shape (x, y, 4), RGBA

        #grn = im[:,:,1]
        # combine all channels. It is not easy to select the best channel
        grn = np.amin(im, axis=2)
        h, w = grn.shape
        # an area on level 1 image that maps to 1333x800 on full resolution image
        step_x, step_y = ( int(self.step_x / ratio), int(self.step_y / ratio))
        rows, cols = math.ceil(h/step_y), math.ceil(w/step_x)
        m = np.zeros(shape=(rows, cols))
        total_area = step_x * step_y
        for i in range(cols):
            x0 = i * step_x
            x1 = x0 + step_x

            if x0 > w:
                break

            for j in range(rows):
                y0 = j * step_y
                y1 = y0 + step_y
                if y0 > h:
                    break

                window = grn[y0:y1, x0:x1]
                tissue_area = window[window<TH]
                r = tissue_area.size / total_area
                m[j, i] = r

        self.grid = m
        return m

    def next(self):
        """ return next region as np array """

        if self.grid.size == 0:
            self.init_emptiness_grid()

        if self.idx == self.size:  # end of slide regieons
            print('end of slide ')
            return np.array([])

        self.cord[0] = self.idx % self.cols   # col
        self.cord[1] = self.idx // self.cols  # row
        self.x, self.y = self.cord[0] * self.step_x, self.cord[1] * self.step_y
        print('Region coordinates (col, row): (%d, %d), topleft(x, y): (%d,%d), %d of %d' % (
            self.cord[0], self.cord[1], self.x, self.y, self.idx, self.size))

        # this region has very little tissue
        if self.grid[self.cord[1], self.cord[0]] < 0.0001:
            print('region is empty')
            self.idx += 1
            return self.next()

        im = self.read_region((self.x, self.y), 0, (self.step_x, self.step_y))
        self.img = im.convert('RGB')
        self.idx += 1
        return self.img

    def init_size2(self):
        w, h = (self.width, self.height)
        rows = math.ceil(h / (self.step_y - self.region_overlap))
        cols = math.ceil(w / (self.step_x - self.region_overlap))
        self.rows, self.cols = rows, cols
        self.size = rows * cols

    def read_ln_region(self, xy, level, wh):
        x = xy[0] + self._l0_offset[0]
        y = xy[1] + self._l0_offset[1]
        return self.read_region((x, y), level, wh)

    def read_l0_region(self, xy, wh):
        return self.read_ln_region(xy, 0, wh)

    def is_empty_region(self, x1, y1, w, h):
        """ there may be many regions(or scenes if Zeiss), in between regions
        there are empty areas. check whether a region is empty """
        if not self.regions:
            return False

        x2, y2 = x1 + w, y1 + h
        for region in self.regions:
            if (min(x2, region['x2']) > max(x1, region['x1']) and
                min(y2, region['y2']) > max(y1, region['y1'])):
                return False

        return True

    def next2(self):
        """ return next region as np array. The regions are overlapping by
        REGION_OVERLAP. The mask will be cut of at REGION_OVERLAP/2 -
        MASK_OVERLAP.

        for example, REGION_OVERLAP is 200, part of the mask that closer to the
        border less than 100 - MASK_OVERLAP will be cut of. The same mask on neighbor
        region will overlap by MASK_OVERLAP, which can be used to join them together
        """
        col, row = self.cord
        self.x =  col * (self.step_x - self.region_overlap) + self.x_offset
        self.y =  row * (self.step_y - self.region_overlap) + self.y_offset

        w, h = (self.width, self.height)
        if self.x >= w:
            col = 0
            row += 1

        if self.y >= h:  # end of slide regieons
            print('end of slide ')
            return np.array([])

        self.x =  col * (self.step_x - self.region_overlap) + self.x_offset
        self.y =  row * (self.step_y - self.region_overlap) + self.y_offset
        print('Region cord (col,row):(%d,%d)/(%d,%d), TL(x, y):(%d,%d), %d/%d' % (
            col, row, self.cols, self.rows, self.x, self.y, self.idx, self.size))

        x = self.x + self._l0_offset[0]
        y = self.y + self._l0_offset[1]

        if self.is_empty_region(x, y, self.step_x, self.step_y):
            self.img = None
        else:
            im = self.read_region((x, y), 0, (self.step_x, self.step_y))
            self.img = im.convert('RGB')

        self.idx += 1
        self.cord = [col+1, row]  # move to the right
        return self.img

    def init_mask_tile(self):
        """ call it before serve mask tiles """

        self._mask_z_t_downsample = TILESIZE
        self._mask_z_overlap = TILEOVERLAP

        # Deep Zoom level
        #z_size = self.level_dimensions[0]
        z_size = (self.width, self.height)
        z_dimensions = [z_size]
        while z_size[0] > 1 or z_size[1] > 1:
            z_size = tuple(max(1, int(math.ceil(z / 2))) for z in z_size)
            z_dimensions.append(z_size)

        # pixel dimension of each level, [(2px,2px), (4px,4px), (8px,8px) ... ]
        self._mask_z_dimensions = tuple(reversed(z_dimensions))
        # Tile
        # _mask_z_t_downsample: tile size 254
        tiles = lambda z_lim: int(math.ceil(z_lim / self._mask_z_t_downsample))
        # tile dimension of each level, starts from zoom out all
        self._mask_t_dimensions = tuple((tiles(z_w), tiles(z_h))
                    for z_w, z_h in self._mask_z_dimensions)

        # Deep Zoom level count
        self._mask_dz_levels = len(self._mask_z_dimensions)

        # Total downsamples for each Deep Zoom level
        # downsample factors (f0, f1, f2, ... ),
        # imgsizeof_this_level * downsample_factor = fullsize_image
        self._mask_dz_factors = tuple(2 ** (self._mask_dz_levels - dz_level - 1)
                    for dz_level in range(self._mask_dz_levels))

    def get_mask_dzi(self, format):
        """Return a string containing the XML metadata for the .dzi file.

        format:    the format of the individual tiles ('png' or 'jpeg')"""
        image = Element('Image', TileSize=str(self._mask_z_t_downsample),
                        Overlap=str(self._mask_z_overlap), Format=format,
                        xmlns='http://schemas.microsoft.com/deepzoom/2008')
        w, h = (self.width, self.height)
        SubElement(image, 'Size', Width=str(w), Height=str(h))
        tree = ElementTree(element=image)
        buf = BytesIO()
        tree.write(buf, encoding='UTF-8')
        return buf.getvalue().decode('UTF-8')

    def get_mask_tile(self, db, lock, level, address, color=True):
        """ Return a numpy array for a tile.

        level:     the Deep Zoom level.
        address:   the address of the tile within the level as a (col, row)
                   tuple.
        color:     use color to difference masks
        """

        return self._ut_get_or_set_tile(db, lock, level, address, color)

    def _ut_is_valid_tile(self, level, address):
        dim = self._mask_t_dimensions[level]
        if address[0] > (dim[0] - 1) or address[1] > (dim[1] - 1):
            return False

        if address[0] < 0 or address[1] < 0:
            return False

        return True

    def _ut_get_or_set_tile(self, db, lock, level, address, color=True):
        tile = self._ut_get_tile(db, level, address)
        if tile is None:
            tile = self._ut_set_tile(db, lock, level, address, color)

        return tile

    def _ut_get_tile(self, db, level, address):
        """ return color tile when available, else binary mask """
        if not self._ut_is_valid_tile(level, address):
            print('not valid: level, address', level, address)
            return None

        sql = 'SELECT tile,btile from Tile WHERE lid=? AND cid=? AND rid=?'
        q = query_db(db, sql, (level, address[0], address[1]), one=True)
        if q and q[0]:
            return blob2mask(q[0])     # np array for multi-channel image

        elif q and q[1]:
            return rleblob2mask(q[1])  # np array for binary mask

        return None

    def _ut_set_tile(self, db, lock, level, address, color=True):
        ''' block if cannot acquire db lock '''

        #print('level %d tile (%d, %d): NOT found in db' %
        #        (level, address[0],address[1]))

        # need to generate new tiles, lock db first
        if lock is not None:
            lock.acquire()
        #print('Get level %d, (%d,%d), db locked' % (
        #                                level, address[0], address[1]))

        # let's check again the tile is not generated while waiting for db lock
        tile = self._ut_get_tile(db, level, address)
        if tile:
            print('tile generated while waiting')
            if lock is not None:
                lock.release()
            return tile

        tilesize = TILESIZE

        top_level = self._mask_dz_levels - 1  # level of full size
        #print('current_level=', level, ', tile dimension ',
        #    self._mask_t_dimensions[level], self._mask_z_dimensions[level])

        if level == top_level:
            self._ut_save_fullsize_tiles(db, top_level, color)
            if lock is not None:
                lock.release()
            return self._ut_get_tile(db, level, address)

        # not top_level, extract 4 tiles from one level above, scale them down
        cid, rid = address

        dim =  self._mask_t_dimensions[level]
        dim1 = self._mask_t_dimensions[level+1]
        # upper level can have same number of columns or rows if zoom level
        # is high
        if dim[0] == dim1[0]:
            cid_up = cid
        else:
            cid_up = cid * 2

        if dim[1] == dim1[1]:
            rid_up = rid
        else:
            rid_up = rid * 2

        if cid_up == 0:
            w_up = TILESIZE * 2 + TILEOVERLAP
            w_me = TILESIZE + TILEOVERLAP
        else:
            w_up = TILESIZE * 2 + TILEOVERLAP2
            w_me = TILESIZE + TILEOVERLAP2

        if rid_up == 0:
            h_up = TILESIZE * 2 + TILEOVERLAP
            h_me = TILESIZE + TILEOVERLAP
        else:
            h_up = TILESIZE * 2 + TILEOVERLAP2
            h_me = TILESIZE + TILEOVERLAP2

        shape = (h_up, w_up, 4) if color else (h_up, w_up)
        canvas = np.zeros(shape, dtype=np.uint8)

        y = 0
        # merge upper level 2x2 tiles
        for r in range(2):
            x = 0
            for c in range(2):
                addr = (cid_up + c, rid_up + r)
                w,h = self._ut_get_mask_tile_info(level+1, addr)
                if not self._ut_is_valid_tile(level+1, addr):
                    x += w - 1
                    continue

                # need release and re-acquire lock
                if lock is not None:
                    lock.release()
                t = self._ut_get_or_set_tile(db, lock, level+1, addr, color)
                if lock is not None:
                    lock.acquire()

                th, tw = (t.shape[0], t.shape[1])
                if color:
                    canvas[y: y+th, x:x+tw] = t
                else:
                    canvas[y: y+th, x:x+tw][t != 0] = 255

                x += w - TILEOVERLAP2

            y += h - TILEOVERLAP2

        canvas = ski_resize(canvas, (h_me, w_me), preserve_range=True)
        if color:
            tile = canvas.astype(np.uint8)
        else:
            # after resize, there are many non-zero pixels in background, shows
            # as vertical strips. Use a large value (100) to cut off these
            # resizing artifact
            tile = np.zeros(canvas.shape, dtype=np.uint8)
            tile[canvas > 100] = 1

        cur = db.cursor()

        if color:
            blob = array2pngblob(tile)
            sql_add = 'INSERT INTO Tile (lid,cid,rid,tile) VALUES(?,?,?,?)'
        else:
            blob = mask2rleblob(tile)
            sql_add = 'INSERT INTO Tile (lid,cid,rid,btile) VALUES(?,?,?,?)'

        print('save 4->1 image into sqlite, level', level, address)
        cur.execute(sql_add, (level, cid, rid, blob))

        if lock is None:
            self.commit_count += 1
            if self.commit_count % MIN_COMMIT == 0:
                db.commit()
        else:
            db.commit()
            cur.close()
            lock.release()

        return tile

    def _ut_get_mask_tile_info(self, level, address):
        w, h = (TILESIZE, TILESIZE)
        dim = self._mask_t_dimensions[level]
        w += TILEOVERLAP if address[0] == 0 else TILEOVERLAP2
        h += TILEOVERLAP if address[1] == 0 else TILEOVERLAP2
        return (w,h)

    def _ut_save_fullsize_tiles(self, db, level, color):
        print('  - enter save fullsize tile')

        # initilize random color for mask
        if self.rndcolors is None:
            rows = query_db(db, 'select id from Mask where is_bad=0', one=False)
            self.rndcolors = {}
            for r in rows:
                if color:
                    """
                    # vary V in green color HSV(120, 100, 100)
                    tmp = colorsys.hsv_to_rgb(0.333333,
                                            np.random.uniform(0.5, 1.0),
                                            np.random.uniform(0.4, 1.0))
                    c = [int(x * 255) for x in tmp]
                    c.append(200)  # alpha
                    self.rndcolors[r[0]] = c
                    """
                    self.rndcolors[r[0]] = list(np.random.randint(0, 255, (3,)))
                    self.rndcolors[r[0]].append(200)
                else:
                    self.rndcolors[r[0]] = [255, 255, 0, 170]  # yellow

        commit_count = 0
        # offset of mask y value on full resolution image to y inside row
        pos_y0 = 0

        # bottom y of current row on full resolution image
        pos_y1 = TILESIZE + TILEOVERLAP  # first row only has bottom overlap

        rid = 0
        nrows = self._mask_t_dimensions[level][1]
        print('top level has %d rows' % nrows)

        sql = ('SELECT id,x,y,w,h,mask from Mask WHERE (y+h-1)>? AND y <?'
               ' AND is_bad=0')
        cur = db.cursor()
        if color:
            sql_add = 'INSERT INTO Tile (lid,cid,rid,tile) VALUES(?,?,?,?)'
        else:
            sql_add = 'INSERT INTO Tile (lid,cid,rid,btile) VALUES(?,?,?,?)'

        while rid < nrows:
            # generate tile row after row
            # a 20x aperio image will have 30k pixels in width, single channel
            # image 256px tall uses 7M memory
            w = self._mask_z_dimensions[level][0]
            h = TILESIZE + TILEOVERLAP if rid == 0 else TILESIZE + TILEOVERLAP2
            shape = (h, w, 4) if color else (h, w)
            canvas = np.zeros(shape, dtype=np.uint8)

            masks = query_db(db, sql, (pos_y0, pos_y1))
            for m in masks:
                #if m['w'] > 1000 or m['h'] > 1000:
                #    continue

                mask = rleblob2mask(m['mask'])

                # translate to coordinates on row strip
                x, y = m['x'], m['y'] - pos_y0
                x1, y1 = x + m['w'], y + m['h']

                # y or y1 may outside of canvas
                if y < 0:
                    mask = mask[y * -1:, :]
                    y = 0

                if y1 >= h:
                    mask = mask[:h - y, :]
                    y1 = h

                if x1 >= w:
                    mask = mask[:, :w - x]
                    x1 = w

                if color:
                    c = self.rndcolors[m['id']]
                    np.copyto(canvas[y: y1, x: x1, 0], c[0], where=mask)
                    np.copyto(canvas[y: y1, x: x1, 1], c[1], where=mask)
                    np.copyto(canvas[y: y1, x: x1, 2], c[2], where=mask)
                    np.copyto(canvas[y: y1, x: x1, 3], c[3], where=mask)
                else:
                    u, counts = np.unique(mask, return_counts=True)
                    canvas[y: y1, x: x1][mask != 0] = 1

            # split the row into overlapping tiles and save to db
            tx0, tx1 = 0, TILESIZE + TILEOVERLAP
            print('  - debug: store level %d tiles -> row %d/%d' % (
                                                        level, rid, nrows))
            for cid in range(self._mask_t_dimensions[level][0]):
                tx0 = 0 if cid == 0 else cid * TILESIZE - TILEOVERLAP2
                tx1 = (tx0 + TILESIZE + TILEOVERLAP if cid == 0
                       else tx0 + TILESIZE + TILEOVERLAP2)
                tile = canvas[0:h, tx0: tx1]
                blob = array2pngblob(tile) if color else mask2rleblob(tile)

                cur.execute(sql_add, (level, cid, rid, blob))
                commit_count += 1
                if commit_count % MIN_COMMIT == 0:
                    db.commit()

            pos_y0 = pos_y1 - TILEOVERLAP2
            pos_y1 = pos_y0 + TILESIZE + TILEOVERLAP2

            rid += 1

        db.commit()
        cur.close()
        print('  - end save fullsize tile')
        # all tiles of top level has been saved to db
        return


# ----------------------- helper ----------------------------
def query_db(db, query, args=(), one=False):
    ''' wrap the db query, fetch into one step '''
    cur = db.execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv


def update_db_schema(db):
    # update Mask table, add polygon column
    print('debug: check and update db schema if necessary')
    sql = ("SELECT count(*) FROM pragma_table_info('Info') "
           " WHERE name='maskfmt'")
    row = query_db(db, sql, one=True)
    if row and row[0] == 0:
        print('update db schema: add column maskfmt(Info) and btile(Tile)')
        db.execute('ALTER TABLE Info ADD COLUMN maskfmt INTEGER;')
        db.execute('ALTER TABLE Tile ADD COLUMN btile BLOB')
        db.commit()

    sql_idx = ("SELECT count(*) from sqlite_master where type='index' "
               "and tbl_name = 'Tile' and name = 'idx_Tile_lid'")
    row = query_db(db, sql_idx, one=True)
    if row and row[0] == 0:
        sql_add_idx = """
CREATE INDEX IF NOT EXISTS idx_Tile_lid on Tile (lid);
CREATE INDEX IF NOT EXISTS idx_Tile_cid on Tile (cid);
CREATE INDEX IF NOT EXISTS idx_Tile_rid on Tile (rid);
    """
        db.executescript(sql_add_idx)
        db.commit()

    upgradedb5_mask_add_cx_cy(db)


def upgradedb1_mask_add_note(db):
    cur = db.cursor()
    try:
        cur.execute('ALTER TABLE Mask ADD COLUMN note TEXT')
        db.commit()
    except sqlite3.OperationalError:
        pass


def upgradedb2_mask_add_mm_seg(db):
    cur = db.cursor()
    try:
        cur.execute('ALTER TABLE Mask ADD COLUMN mm_seg BLOB')
        db.commit()
    except sqlite3.OperationalError:
        pass


def upgradedb3_mask_add_mm_pas(db):
    cur = db.cursor()
    try:
        cur.execute('ALTER TABLE Mask ADD COLUMN mm_pas BLOB')
        db.commit()
    except sqlite3.OperationalError:
        pass


def upgradedb4_mask_add_is_bad(db):
    cur = db.cursor()
    try:
        cur.execute('ALTER TABLE Mask ADD COLUMN is_bad INTEGER DEFAULT 0')
        db.commit()
    except sqlite3.OperationalError:
        pass


def upgradedb5_mask_add_nuclei_and_opened(db):
    # mask_nuclei BLOB,         /* nuclei in glm */
    # mm_opened BLOB,           /* mesangial matrix after open */
    cur = db.cursor()
    try:
        cur.execute('ALTER TABLE Mask ADD COLUMN mask_nuclei BLOB')
        cur.execute('ALTER TABLE Mask ADD COLUMN mm_opened BLOB')
        db.commit()
    except sqlite3.OperationalError:
        pass

def upgradedb5_mask_add_cx_cy(db):
    sql = ("SELECT count(*) FROM pragma_table_info('Mask') "
           " WHERE name='cx'")
    row = query_db(db, sql, one=True)
    if row and row[0] == 0:
        print('update db schema: add column cx, cy to 4 (date type INTEGER) to Mask')
        db.executescript('''
ALTER TABLE Mask ADD COLUMN cx INTEGER;
ALTER TABLE Mask ADD COLUMN cy INTEGER;
CREATE INDEX IF NOT EXISTS idx_Mask_cx on Mask (cx);
CREATE INDEX IF NOT EXISTS idx_Mask_cy on Mask (cy);
CREATE INDEX IF NOT EXISTS idx_Mask_x on Mask (x);
CREATE INDEX IF NOT EXISTS idx_Mask_y on Mask (y);''')
        db.commit()

    db.execute('''UPDATE Mask set cx=cast((x+w*0.5+0.5) as integer),
                                  cy=cast((y+h*0.5+0.5) as integer);''')
    db.commit()


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


def blob2mask(blob):
    stream = BytesIO(blob)
    mask = Image.open(stream)
    return np.asarray(mask)


def array2pngblob(im):
    stream = BytesIO()
    tmp = Image.fromarray(im)
    tmp.save(stream, format='PNG', optimize=True)
    blob = stream.getvalue()
    return blob


def array2jpgblob(im, quality=85):
    """
    for glm pas image, jpg is smaller than jpeg2000, 90KB vs 290KB
    Format        png       jpeg     jpeg2000
    -----------------------------------------
    Size(KB)      466         91          292
    """

    stream = BytesIO()
    tmp = Image.fromarray(im)
    tmp.save(stream, format='JPEG', optimize=True, quality=quality)
    #tmp.save(stream, format='JPEG2000', optimize=True, quality=85)
    blob = stream.getvalue()
    return blob


def blob2img(blob):
    stream = BytesIO(blob)
    return Image.open(stream)


