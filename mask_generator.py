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

# Root directory of the project
ROOT_DIR = os.path.abspath('')

INPUT_DIR = '/media/RAIDONE/radice'
OUTPUT_DIR = '/media/RAIDONE/radice/STRUCT2DEPTH'
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
                        choices=['OXFORD'])
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
    print('-> Processing', folder)
    if not os.path.isdir(os.path.join(OUTPUT_DIR, dataset, folder, 'masks')):
        os.mkdir(os.path.join(OUTPUT_DIR, dataset, folder, 'masks'))
    if not os.path.isdir(os.path.join(OUTPUT_DIR, dataset, folder, 'masks', subfolder)):
        os.mkdir(os.path.join(OUTPUT_DIR, dataset, folder, 'masks', subfolder))

    for file in files:
        image = cv2.imread(file)
        image = image[CROP_AREA[1]:CROP_AREA[3], :, :]

        # Run detection
        results = model.detect([image], verbose=1)

        r = results[0]
        masks = r['masks'].copy()
        mask = np.zeros((masks.shape[0], masks.shape[1]))

        # list of ids we want
        mask_ids = [1, 2, 3, 4, 6, 8]

        for k in range(masks.shape[2]):
            if r['class_ids'][k] in mask_ids and r['scores'][k] > 0.98:
                it = np.nditer(masks[:, :, k], flags=['multi_index'])
                for x in it:
                    if x == True:
                        mask[it.multi_index[0], it.multi_index[1]] = 255

        basename = os.path.basename(file).split('.')[0]
        mask_path = os.path.join(OUTPUT_DIR, dataset, folder, 'masks', subfolder, '{}{}.{}'.format(basename, '-fseg', 'png'))
        print(file)
        print(mask_path)
        cv2.imwrite(mask_path, mask)


def mask_generator_gpu(model, files, dataset, folder, subfolder):
    print('-> Processing', folder)
    if not os.path.isdir(os.path.join(OUTPUT_DIR, dataset, folder, 'masks')):
        os.mkdir(os.path.join(OUTPUT_DIR, dataset, folder, 'masks'))
    if not os.path.isdir(os.path.join(OUTPUT_DIR, dataset, folder, 'masks', subfolder)):
        os.mkdir(os.path.join(OUTPUT_DIR, dataset, folder, 'masks', subfolder))

    batch_size = 1

    images = []
    file_names = []
    i = 1
    for file in files:

        image = cv2.imread(file)
        image = image[CROP_AREA[1]:CROP_AREA[3], :, :]

        images.append(image)
        file_names.append(file)

        if i % batch_size == 0 and i != 0:
            print(len(images))
            idx = 0
            # Run detection
            results = model.detect(images, verbose=1)

            for r in results:
                masks = r['masks'].copy()
                mask = np.zeros((masks.shape[0], masks.shape[1]))

                # list of ids we want
                mask_ids = [1, 2, 3, 4, 6, 8]

                for i in range(masks.shape[0]):
                    for j in range(masks.shape[1]):
                        for k in range(masks.shape[2]):
                            # we want to be sure the mask is correct
                            if r['class_ids'][k] in mask_ids:
                                if (masks[i, j, k] == True) and (r['scores'][k] > 0.96):
                                    mask[i, j] = 255

                basename = os.path.basename(file_names[idx]).split('.')[0]
                mask_path = os.path.join(OUTPUT_DIR, dataset, folder, 'masks', subfolder, '{}{}.{}'.format(basename, '-fseg', 'png'))
                print(file_names[idx])
                print(mask_path)
                cv2.imwrite(mask_path, mask)

                idx += 1

            images = []
            file_names = []

        i += 1


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

    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle',
                   'bus', 'truck']

    # Directory of images to run detection on
    if dataset == 'OXFORD':
        left_path = os.path.join(INPUT_DIR, dataset, folder, 'processed', 'stereo', 'left')
        right_path = os.path.join(INPUT_DIR, dataset, folder, 'processed', 'stereo', 'right')

        left_files = glob.glob(left_path + '/*.jpg')
        right_files = glob.glob(right_path + '/*.jpg')

        left_files = sorted(left_files)
        right_files = sorted(right_files)

        mask_generator(model=model, files=left_files, dataset=dataset, folder=folder, subfolder='left')
        mask_generator(model=model, files=right_files, dataset=dataset, folder=folder, subfolder='right')

        #mask_generator_gpu(model=model, files=left_files, dataset=dataset, folder=folder, subfolder='left')

if __name__ == '__main__':
    args = parse_args()
    main(args)