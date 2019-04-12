# KITTI Detection

A project for KITTI detection challenge (Extend from [mmdetection](https://github.com/open-mmlab/mmdetection))!

## Installation

Required environment:

- cuda9.0+
- python3
- pytorch0.40+

Install our python3.7 experiment with conda .yml file with scripts:

```shell

# python3.7 cuda9.0
conda env create -f py3.7-torch0.4.1-cuda9.0.yml

# or python3.7 cuda9.2
conda env create -f py3.7-torch0.4.1-cuda9.2.yml

```

Also you can install python3.6 environment with pytorch0.40+ manually.


At last, you need to complie some components for installation with script:

```shell
./complie
```

## Get Started

- KITTI Detection Dataset prepared

Go to [KITTI Detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark) and 
download [left color images of object data set (12 GB)](http://www.cvlibs.net/download.php?file=data_object_image_2.zip)
to your datasets directory.

Then run script under `tools/convert_datasets/kitti_detection.py` to format data for loading
 (NOTE: you have to indicate datasets directory with flag in `kitti_detection.py`).

Edit config in `configs/faster_rcnn_r50_fpn_kitti.py` to  indicate your datasets directory.

- Try with Faster-RCNN

Play Faster-RCNN with scripts:

```shell
python tools/train.py configs/faster_rcnn_r50_fpn_kitti.py --work_dir kitti_full --validate

```

Then you can play tensorboard server to your tf_log directory in your work_dir to visual training and testing curve.
(NOTE: you have to install tensorboard and tensorflow for tensorboard server which is not supported in py3.7)


> Thanks for tools [mmlab](https://github.com/open-mmlab) provide!
