import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import argparse
import cv2
import glob
# run time
import timeit
# time format
import time
import pickle

# Root directory of the project
ROOT_DIR = os.path.abspath('')

DIR = '/media/RAIDONE/radice/datasets/'
CROP_AREA = [0, 360, 1280, 730]

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

def parse_args():
    parser = argparse.ArgumentParser(description='Data generator for depth-and-motion-learning')
    parser.add_argument('--folder', type=str,
                        help='folder containing files',
                        required=True)
    parser.add_argument('--dataset', type=str,
                        help='dataset',
                        choices=['oxford'])
    return parser.parse_args()

# Configurations
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the CocoConfig class in
# coco.py.
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the CocoConfig class and
# override the attributes you need to change.

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def mask_generator(model, files, dataset, folder, subfolder):
    # start timer
    start = timeit.default_timer()
    current_seg = start

    if not os.path.isdir(os.path.join(DIR, dataset, folder, 'rcnn-masks')):
        os.mkdir(os.path.join(DIR, dataset, folder, 'rcnn-masks'))
    if not os.path.isdir(os.path.join(DIR, dataset, folder, 'rcnn-masks', subfolder)):
        os.mkdir(os.path.join(DIR, dataset, folder, 'rcnn-masks', subfolder))

    print('-> Processing', subfolder, 'camera frames')
    print('-> Save path', os.path.join(DIR, dataset, folder, 'rcnn-masks', subfolder))

    count = 0
    for file in files:
        image = cv2.imread(file)
        image = image[CROP_AREA[1]:CROP_AREA[3], :, :]

        # Run detection, verbose 0 no print on screen
        results = model.detect([image], verbose=0)

        r = results[0]
        # masks = r['masks'].copy()
        # mask = np.zeros((masks.shape[0], masks.shape[1]))

        # CASO DOVE PRENDO SOLO GLI ID CHE VOGLIO
        # list of ids we want
        # mask_ids = [1, 2, 3, 4, 6, 8]
        #
        # for k in range(masks.shape[2]):
        #     if r['class_ids'][k] in mask_ids and r['scores'][k] > 0.98:
        #         it = np.nditer(masks[:, :, k], flags=['multi_index'])
        #         for x in it:
        #             if x == True:
        #                 mask[it.multi_index[0], it.multi_index[1]] = 255

        # CASO DOVE PRENDO TUTTI GLI ID
        # for k in range(masks.shape[2]):
        #     if r['scores'][k] > 0.90:
        #         it = np.nditer(masks[:, :, k], flags=['multi_index'])
        #         for x in it:
        #             if x == True:
        #                 mask[it.multi_index[0], it.multi_index[1]] = 255

        #     mask_path = os.path.join(DIR, dataset, folder, 'rcnn-masks', subfolder, '{}{}.{}'.format(basename, '-fseg', 'png'))
        #     cv2.imwrite(mask_path, mask)

        # Salvo in un file .pkl l'output della rete neurale con score > 80
        dict = {}
        for k in range(len(r['scores'])):
            if r['scores'][k] < 0.80:
                dict['rois'] = np.delete(r['rois'], k, axis=0)
                dict['scores'] = np.delete(r['scores'], k)
                dict['class_ids'] = np.delete(r['class_ids'], k)
                dict['masks'] = np.delete(r['masks'], k, axis=2)

        basename = os.path.basename(file).split('.')[0]
        dict_save_path = os.path.join(DIR, dataset, folder, 'rcnn-masks', subfolder, '{}.pkl'.format(basename))
        with open(dict_save_path, 'wb') as handle:
            pickle.dump(dict, handle)

        if (count % 1000 == 0) and (count != 0):
            print('->', count, 'Done')
            # segment run time
            stop_seg = timeit.default_timer()
            seg_run_time = int(stop_seg - current_seg)
            print('-> Segment run time:', time.strftime('%H:%M:%S', time.gmtime(seg_run_time)))
            current_seg += seg_run_time

        count += 1

    # partial run time
    stop = timeit.default_timer()
    partial_run_time = int(stop - start)
    print('-> Partial run time:', time.strftime('%H:%M:%S', time.gmtime(partial_run_time)))


def main(args):
    folder = args.folder
    dataset = args.dataset

    config = InferenceConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    ## Class Names
    # The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets
    # assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1
    # and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes
    # associated with class IDs 70 and 72, but not 71.
    # To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset```
    # class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our
    # ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78
    # (different from COCO). Keep that in mind when mapping class IDs to class names.
    # To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.

    # Load COCO dataset
    # dataset = coco.CocoDataset()
    # dataset.load_coco(COCO_DIR, "train")
    # dataset.prepare()
    # Print class names
    # print(dataset.class_names)

    # We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class
    # names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2,
    # ...etc.)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')

    # class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    #                'bus', 'train', 'truck', 'boat', 'traffic light',
    #                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    #                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    #                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    #                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    #                'kite', 'baseball bat', 'baseball glove', 'skateboard',
    #                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    #                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    #                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    #                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    #                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    #                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    #                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    #                'teddy bear', 'hair drier', 'toothbrush']

    # Directory of images to run detection on
    if dataset == 'oxford':
        left_path = os.path.join(DIR, dataset, folder, 'stereo', 'left')
        # right_path = os.path.join(DIR, dataset, folder, 'stereo', 'right')

        left_files = glob.glob(left_path + '/*.png')
        # right_files = glob.glob(right_path + '/*.png')

        left_files = sorted(left_files)
        # right_files = sorted(right_files)

        mask_generator(model=model, files=left_files, dataset=dataset, folder=folder, subfolder='left')
        # mask_generator(model=model, files=right_files, dataset=dataset, folder=folder, subfolder='right')


if __name__ == '__main__':
    # start timer
    start = timeit.default_timer()

    args = parse_args()
    main(args)

    # stop timer
    stop = timeit.default_timer()

    # total run time
    total_run_time = int(stop - start)
    print('-> Total run time:', time.strftime('%H:%M:%S', time.gmtime(total_run_time)))