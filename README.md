# Mask RCNN

## INSTALLAZIONE
 
### Requirements 

Conda env:
- python=3.6
- jupyter-lab   (non su server)    conda install -c conda-forge jupyterlab
- anaconda pip                     conda install -c anaconda pip

With pip:
- tensorflow-gpu==1.13.1
- tensorflow==1.13.1
- keras==2.0.8
- scikit-image
- imgaug
- Cython
- pycocotools
- h5py==2.10.0

### Setup

```shell
python setup.py install
```

## Training Dataset Generator

Generazione delle maschere per il dataset di training.

Salvo il dizionario contenente le maschere come file compresso .npz per poi poterlo usare per creare 
le maschere con la soglia di score che preferisco.

### Oxford

- crop immagini: [0, 360, 1280, 730]
- test crop immagini: [0, 200, 1280, 810] (più grandi magari meglio)
- uso solo frames della camera di sinistra

```shell
python Mask_RCNN/mask_generator_oxford.py --folder --dataset
```

### Kitti

```shell
python Mask_RCNN/mask_generator_kitti.py
```