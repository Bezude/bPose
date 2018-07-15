import os
import sys
import cv2
import numpy as np
import argparse as ap
# import cnnArchs
# from builtins import FileExistsError
from pycocotools.coco import COCO
# import pycocotools.mask as msk
import matplotlib.pyplot as plt
from genCrowdMasks import genCrowdMasks


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


def test(dataDir='', dataType='train2014', imgId=None):
    if not dataDir:
        dataDir = os.path.dirname(os.path.realpath(sys.argv[0]))
    anFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)
    cc = COCO(anFile)
    catIds = cc.getCatIds(catNms=['person'])
    if (len(catIds) == 0):
        print('No files with "person" category in annotations.')
        return
    if (imgId is None):
        imgIds = cc.getImgIds(catIds=catIds)
    else:
        imgIds = [imgId]
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


def main():
    parser = ap.ArgumentParser(description='2D Pose Detection w/ PyTorch')
    parser.add_argument('-d', '--data', dest='data_dir', help='path to dataset')
    parser.add_argument('-t', '--test', dest='test_dir',
                        help='directory for annotation testing')
    parser.add_argument('-i', '--image', dest='test_img', type=int,
                        help='specific id of an image to view annotations of')
    parser.add_argument('-s', '--dataset', dest='dataset',
                        help='example: train2017')
    parser.add_argument('-l', '--limit', dest='limit', type=int,
                        help='only create this many masks (for testing)')
    args = parser.parse_args()
    # getMeanAndStdOfImgDir(os.path.join(args.dir, 'train'))
    # myNet = cnnArchs.PAF_Cao_et_al()
    if (args.test_dir is not None):
        try:
            test(dataDir=args.test_dir, imgId=args.test_img)
        except AssertionError:
            print('Test images did not load properly. Check given directory ' +
                  'is correct. It should end with a slash.')
    n_masks = args.limit if args.limit is not None else 0
    if (args.data_dir is not None):
        if (args.dataset is not None):
            genCrowdMasks(args.data_dir, dataType=args.dataset, limit=n_masks)
        else:
            genCrowdMasks(args.data_dir, dataType='train2017', limit=n_masks)


if __name__ == '__main__':
    main()
