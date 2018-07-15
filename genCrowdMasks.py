import os
import cv2
import sys
import numpy as np
from builtins import FileExistsError
from pycocotools.coco import COCO
import pycocotools.mask as msk


def genCrowdMasks(dataDir, dataType='train2017', limit=0, overwrite=False):
    if not dataDir:
        dataDir = dataDir = os.path.dirname(os.path.realpath(sys.argv[0]))
    maskDir = os.path.join(dataDir, 'masks/', dataType, '')
    try:
        os.makedirs(maskDir)
    except FileExistsError:
        print('masks directory already exists')
    anFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)
    cc = COCO(anFile)
    img_ids = cc.getImgIds(catIds=cc.getCatIds(['person']))
    if (len(img_ids) == 0):
        print('Annotation file contains no "person" images')
        return
    if (limit == 0):
        limit == len(img_ids)
    _genMasks(img_ids, cc, maskDir, limit, overwrite)
    print('{} masks created in total'.format(limit))


def _genMasks(img_ids, cc, maskDir, limit, overwrite):
    # I don't completely understand why the tuple and map are needed
    # but without them fillPoly throws:
    #   TypeError: Scalar value for argument 'color' is not numeric
    black = tuple(map(int, np.uint8(np.array([0, 0, 0]))))
    for i in range(limit):
        if (i % 100 == 0):
            print('{} masks created so far'.format(i))
        anns = cc.loadAnns(cc.getAnnIds(imgIds=img_ids[i]))
        img = cc.loadImgs(img_ids[i])[0]
        f_name = os.path.join(maskDir, img['file_name'][:-3] + 'png')
        if (not overwrite and os.path.isfile(f_name)):
            continue
        crowdMask = np.ones((img['height'], img['width'], 3), dtype=np.uint8)
        crowdMask = crowdMask * np.array([255, 255, 255], dtype=np.uint8)
        for ann in anns:
            if (ann['iscrowd']):
                mask = msk.frPyObjects([ann['segmentation']],
                                       img['height'], img['width'])
                mask = msk.decode(mask)
                mask ^= 1
                crowdMask *= mask
            elif (ann['num_keypoints'] == 0):
                try:
                    pts = np.reshape(ann['segmentation'][0], (-1, 2))
                except KeyError:
                    print(ann['segmentation'])
                pts = pts.astype(int)
                cv2.fillPoly(crowdMask, [pts], black)
        cv2.imwrite(f_name, crowdMask)
