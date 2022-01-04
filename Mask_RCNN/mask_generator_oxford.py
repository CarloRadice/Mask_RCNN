import os
import sys
import numpy as np
import cv2
import glob
import timeit
import time
import argparse

# Root directory of the project
ROOT_DIR = os.path.abspath('')

DIR = '/media/RAIDONE/radice/datasets/oxford'
CROP_AREA = [0, 200, 1280, 810]

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


def parse_args():
    parser = argparse.ArgumentParser(description='Data generator for depth-and-motion-learning')
    parser.add_argument('--folder', type=str,
                        help='folder containing files',
                        required=True)
    return parser.parse_args()


def main(args):
    folder = args.folder

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
    print('Processing sequence', folder)
    path_to_lr_folders = os.path.join(DIR, folder, 'stereo')
    for lr in glob.glob(path_to_lr_folders + '/*'):

        print('Processing folder', os.path.basename(lr))

        mask_lr_path = os.path.join(DIR, folder, 'rcnn-masks-classes', os.path.basename(lr))
        if not os.path.exists(mask_lr_path):
            os.makedirs(mask_lr_path)

        for file in glob.glob(lr + '/*'):

            basename = os.path.basename(file).split('.')[0]
            dict_save_path = os.path.join(mask_lr_path, '{}'.format(basename))

            if os.path.isfile(dict_save_path + '.npz'):
                continue

            image = cv2.imread(file)
            image = image[CROP_AREA[1]:CROP_AREA[3], :, :]

            # Run detection, verbose 0 no print on screen
            results = model.detect([image], verbose=0)

            r = results[0]

            # Creazione dizionario maschera
            # Ad ogni cella viene associato il valore di score se presente
            # Ad ogni cella viene associata la classe di appartenenza
            dict = {}
            dict['score_mask'] = np.zeros([r['masks'].shape[0], r['masks'].shape[1]], dtype=np.uint8)
            dict['class_ids'] = np.zeros([r['masks'].shape[0], r['masks'].shape[1]], dtype=np.uint8)
            for i in range(r['masks'].shape[0]):
                for j in range(r['masks'].shape[1]):
                    for k in range(r['masks'].shape[2]):
                        if r['masks'][i, j, k] == True:
                            dict['score_mask'][i, j] = np.floor(r['scores'][k] * 100)
                            dict['class_ids'][i, j] = r['class_ids'][k]

            # Salvo il dizionario con compressione
            np.savez_compressed(dict_save_path, dict)


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