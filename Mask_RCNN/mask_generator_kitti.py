import os
import sys
import numpy as np
import cv2
import glob
import timeit
import time

# Root directory of the project
ROOT_DIR = os.path.abspath('')

INPUT_DIR = '/media/RAIDONE/radice/datasets/kitti/data'
OUTPUT_DIR = '/media/RAIDONE/radice/datasets/kitti/mask-rcnn-classes'

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


def mask_generator(model, files, save_path):
    for file in files:
        image = cv2.imread(file)

        basename = os.path.basename(file).split('.')[0]
        dict_save_path = os.path.join(save_path, '{}'.format(basename))

        # Controllo che il file .npz non sia già presente
        # Evito di ricreare files che sono identici
        if not(os.path.isfile(dict_save_path)):
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


def main():
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

    for d in glob.glob(INPUT_DIR + '/*/'):
        date = d.split('/')[-2]

        if not os.path.exists(os.path.join(OUTPUT_DIR, date)):
            os.makedirs(os.path.join(OUTPUT_DIR, date))

        for d2 in glob.glob(d + '*/'):
            seqname = d2.split('/')[-2]
            print('Processing sequence', seqname)

            half_path = os.path.join(OUTPUT_DIR, date, seqname)
            if not os.path.exists(half_path):
                os.mkdir(half_path)

            start_seq = timeit.default_timer()

            for subfolder in ['image_02/data', 'image_03/data']:
                full_path = os.path.join(half_path, subfolder.replace('/data', ''))
                if not os.path.exists(full_path):
                    os.mkdir(full_path)

                print(full_path)

                folder = d2 + subfolder
                files = glob.glob(folder + '/*.png')
                files = [file for file in files if not 'disp' in file and not 'flip' in file and not 'seg' in file]
                files = sorted(files)

                mask_generator(model=model, files=files, save_path=full_path)

            stop_seq = timeit.default_timer()
            # sequence run tume
            seq_run_time = int(stop_seq - start_seq)
            print('-> Total run time:', time.strftime('%H:%M:%S', time.gmtime(seq_run_time)))


if __name__ == '__main__':
    # start timer
    start = timeit.default_timer()

    main()

    # stop timer
    stop = timeit.default_timer()

    # total run time
    total_run_time = int(stop - start)
    print('-> Total run time:', time.strftime('%H:%M:%S', time.gmtime(total_run_time)))