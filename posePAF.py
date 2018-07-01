import os
import sys
import cv2
# import skimage.io as io
import numpy as np
import argparse as ap
# import matplotlib.pyplot as plt
# import cnnArchs
sys.path.append('./pose/cocoapi/PythonAPI/')
from pycocotools.coco import COCO
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


def test(dataDir=''):
    if not dataDir:
        dataDir = os.path.dirname(os.path.realpath(sys.argv[0]))
    dataType = 'train2014'
    anFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)
    cc = COCO(anFile)
    catIds = cc.getCatIds(catNms=['person'])
    if (len(catIds) == 0):
        return
    imgIds = cc.getImgIds(catIds=catIds)
    anns = cc.loadAnns(cc.getAnnIds(imgIds=imgIds))
    anns = [i for i in anns if 'keypoints' in i]
    if (len(anns) == 0):
        return
    # print(anns[0])
    img_fn = 'COCO_{}_{:012d}.jpg'.format(dataType, anns[0]['image_id'])
    imgIds = anns[0]['image_id']
    anns = cc.loadAnns(cc.getAnnIds(imgIds=imgIds))
    fullpath = '{}{}/{}'.format(dataDir, dataType, img_fn)
    img = cv2.imread(fullpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    cc.showAnns(anns)
    plt.show()


def main():
    parser = ap.ArgumentParser(description='2D Pose Detection w/ PyTorch')
    parser.add_argument('-dir', metavar='DIR', help='path to dataset')
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

if __name__ == '__main__':
    main()
