A minimal tensorflow implementation of YOLOv3, with support for training, inference and evaluation.

## Installation
--------------------
Install requirements and download pretrained weights

```
$ pip3 install -r ./docs/requirements.txt
$ wget https://pjreddie.com/media/files/yolov3.weights
```

## Quick start
--------------------
In this part, we will use pretrained weights to make predictions on both image and video.

```
$ python image_demo.py
$ python video_demo.py # if use camera, set video_path = 0
```

<p align="center">
    <img width="100%" src="https://user-images.githubusercontent.com/30433053/68088581-9255e700-fe9b-11e9-8672-2672ab398abe.jpg" style="max-width:100%;">
    </a>
</p>

## Train yymnist
--------------------

<p align="center">
    <img width="70%" src="https://user-images.githubusercontent.com/30433053/68088705-90d8ee80-fe9c-11e9-8e61-589fdc45fe60.png" style="max-width:70%;">
    </a>
</p>

you can train it and then evaluate your model

```
$ python train.py
$ tensorboard --logdir ./data/log
$ python test.py
$ cd ../mAP
$ python main.py        # Detection images are expected to save in `YOLOV3/data/detection`
```
Track training progress in Tensorboard and go to http://localhost:6006/

```
$ tensorboard --logdir ./data/log
```
<p align="center">
    <img width="100%" src="https://user-images.githubusercontent.com/30433053/68088727-db5a6b00-fe9c-11e9-91d6-555b1089b450.png" style="max-width:100%;">
    </a>
</p>