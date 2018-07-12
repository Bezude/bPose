import os
import sys
import cv2
import numpy as np
import argparse as ap
# import cnnArchs
from builtins import FileExistsError
from pycocotools.coco import COCO
import pycocotools.mask as msk
import matplotlib.pyplot as plt


def getMeanAndStdOfImgDir(path_to_dir):
    included_extensions = [
        '.jpg', '.bmp', '.png', '.gif', '.tif', '.jpeg', '.tiff']
    file_names = [fn for fn in os.listdir(path_to_dir)
                  if fn.endswith(tuple(included_extensions))]
    im_stats = np.zeros(shape=(len(file_names), 4))
    for idx, fn in enumerate(file_names):
        im = cv2.imread(os.path.join(path_to_dir, fn))
        im_stats[idx] = np.append(np.sum(im, (0, 1), dtype=np.float),
                                  im.shape[0]*im.shape[1])
    sums = np.sum(im_stats, 0)
    pixel_count = sums[3]
    means = sums[:-1] / pixel_count
    print(pixel_count)
    std = np.zeros(shape=(len(file_names), 3))
    for idx, fn in enumerate(file_names):
        im = cv2.imread(os.path.join(path_to_dir, fn))
        std[idx] = np.sum((im - means)**2, (0, 1))
    std = np.sqrt(np.sum(std, 0) / pixel_count)
    print(means, std)
    return np.append(means, std)


def test(dataDir='', dataType='train2014'):
    if not dataDir:
        dataDir = os.path.dirname(os.path.realpath(sys.argv[0]))
    anFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)
    cc = COCO(anFile)
    catIds = cc.getCatIds(catNms=['person'])
    if (len(catIds) == 0):
        print('No files with "person" category in annotations.')
        return
    imgIds = cc.getImgIds(catIds=catIds)
    anns = cc.loadAnns(cc.getAnnIds(imgIds=imgIds))
    anns = [i for i in anns if 'keypoints' in i]
    if (len(anns) == 0):
        print('No annotations contain keypoints.')
        return
    img_fn = 'COCO_{}_{:012d}.jpg'.format(dataType, anns[0]['image_id'])
    imgIds = anns[0]['image_id']
    anns = cc.loadAnns(cc.getAnnIds(imgIds=imgIds))
    for ann in anns:
        print('num_keypoints: {}'.format(ann['num_keypoints']) +
              '  iscrowd: {}'.format(ann['iscrowd']))
    fullpath = '{}{}/{}'.format(dataDir, dataType, img_fn)
    img = cv2.imread(fullpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    cc.showAnns(anns)
    plt.show()


def genCrowdMasks(dataDir, dataType='train2017'):
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
    green = tuple(map(int, np.uint8(np.array([0, 255, 0]))))
    for i in range(1):
        anns = cc.loadAnns(cc.getAnnIds(imgIds=img_ids[i]))
        img = cc.loadImgs(img_ids[i])[0]
        falseMask = np.zeros((img['height'], img['width'], 3), dtype=np.uint8)
        # falseMask = np.asfortranarray(falseMask)
        # crowdMask = msk.encode(falseMask)
        print('{}:image {} has {} annotations'.format(i, img_ids[i], len(anns)))
        for ann in anns:
            if (ann['iscrowd']):
                mask = msk.frPyObjects([ann['segmentation']],
                                       img['height'], img['width'])
                mask = msk.decode(mask)
                mask = mask * np.array([255, 0, 0], dtype=np.uint8)
                falseMask = np.maximum(falseMask, mask)
            elif (ann['num_keypoints'] == 0):
                buf = np.zeros((img['height'], img['width'], 3), dtype=np.uint8)
                try:
                    pts = np.reshape(ann['segmentation'][0], (-1, 2))
                except KeyError:
                    print(ann['segmentation'])
                pts = pts.astype(int)
                cv2.fillPoly(buf, [pts], green)
                np.maximum(falseMask, buf, out=falseMask)
        plt.imshow(falseMask)
        # plt.axis('off')
    plt.show()


def main():
    parser = ap.ArgumentParser(description='2D Pose Detection w/ PyTorch')
    parser.add_argument('-d', '--data', dest='data_dir', help='path to dataset')
    parser.add_argument('-t', '--test', dest='test_dir',
                        help='directory for annotation testing')
    args = parser.parse_args()
    # getMeanAndStdOfImgDir(os.path.join(args.dir, 'train'))
    # myNet = cnnArchs.PAF_Cao_et_al()
    if (args.test_dir is not None):
        try:
            test(dataDir=args.test_dir)
        except AssertionError:
            print('Test images did not load properly. Check given directory ' +
                  'is correct. It should end with a slash.')
    if (args.data_dir is not None):
        genCrowdMasks(args.data_dir)


if __name__ == '__main__':
    main()
