# MNasNet
[Keras (Tensorflow) Implementation (Eager execution)](https://github.com/Shathe/Semantic-Segmentation-Tensorflow-Eager/blob/master/MnasnetEager.py) of a modification of MNasNet for semantic segmentation (Mnasnet-FC) and an example for training and evaluating it on the Camvid dataset.

Mnasnet paper: [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/pdf/1807.11626.pdf)

## Requirement
* Python 2.7+
* Tensorflow-gpu 1.10
* opencv
* imgaug

## Train it with eager execution
Train the [FC-MNasNet model](https://github.com/Shathe/Semantic-Segmentation-Tensorflow-Eager/blob/master/MnasnetEager.py) on the Camvid dataset! just execute:
```
python train_eager.py
```

![alt text](https://github.com/Shathe/MNasNet-Keras-Tensorflow/raw/master/mnasnet.png)
