# Mask RCNN

## INSTALLAZIONE
 
### Requirements 
-- python=3.6
-- jupyter-lab   (non su server)    conda install -c conda-forge jupyterlab
-- anaconda pip                     conda install -c anaconda pip
-- pip install tensorflow-gpu==1.13.1
-- pip install tensorflow==1.13.1
-- pip install keras==2.0.8
-- pip install scikit-image
-- pip install imgaug
-- pip install Cython
-- pip install pycocotools
-- pip install h5py==2.10.0

### Setup
```shell
python setup.py install
```

## Training Dataset Generator
Generazione delle maschere per il dataset di training.

```shell
python Mask_RCNN/mask_generator.py --folder --dataset
```

### Oxford
- crop immagini: [0, 360, 1280, 730]


### Kitti