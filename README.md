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

```shell
python Mask_RCNN/mask_generator.py --folder --dataset
```

Lo score che misura l'accuracy delle classi deve essere almeno 0.98 per avere la 
certezza che quell'oggetto appartenga alla classe corretta.

Di conseguenza, alcuni oggetti potrebbero non essere rappresentati nella maschera.

### Oxford
- crop immagini: [0, 360, 1280, 730]
- uso solo frames della camera di sinistra

### Kitti
