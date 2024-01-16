## BEVTracking

### Introduction

### Data

Standford Drone Dataset (SDD) is used in this project. The dataset can be downloaded from [here](https://cvgl.stanford.edu/projects/uav_data/).

data format:
```
├── SDD
│   ├── annotations
│   │   ├── bookstore
│   │   │   ├── video0
│   │   │   │   ├── annotations.txt
│   │   │   │   ├── reference.jpg
│   │── videos
│   │   ├── bookstore
│   │   │   ├── video0
│   │   │   │   ├── video.mov
```

annotations.txt format:
```
annotations format:
Track ID, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label
label categories ID map：
Biker：0
Pedestrian：1
Skater：2
Cart：3
Car：4
Bus：5
```
in this project, we only evaluate the performance of pedestrian detection
so we map the biker, skater to pedestrian.

there are some mistake in the annotations:
1. some boxes are out of the image
2. in some moment, one object may have two categories, like pedestrian and biker. 
3. some boxes are fake boxes which are not the real objects
4. some boxes look like waiting for the people to come in
5. when people leave the image, the box is still there, it may be common in the mot task, but not in the detection task.

We filter the dataset and remove the above mistakes.

shown as below:

<video src="figures/ori_vis.mp4" controls="controls" width="500" height="300"></video>


### Installation

bevtracking requires the following dependencies (tested on Windows 11 and Ubuntu 18.04):

**step.1** install paddle
    
```bash
python -m pip install paddlepaddle-gpu==2.5.2.post116 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

**step.2** install requirements

```bash
pip install -r requirements.txt
```

### Running

**step.1** download the dataset and put it in the dataset folder

```
├── project
│   ├── dataset
│   │   ├── standford_campus
│   │   │   ├── videos
│   │   │   ├── annotations
```

**step.2** run the following command to prepare the dataset

```bash
python tools/create_data.py
```

**step.3** run the following command to train the model

For single GPU training:
```bash
python tools/train.py -c configs/smalldet/ppyoloe_plus_sod_crn_l_40e_coco.yml --eval --use_vdl
```
For multi GPU training:
```bash
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" tools/train.py -c configs/smalldet/ppyoloe_plus_sod_crn_l_40e_coco.yml --eval --use_vdl
```
In our config setting, we use 8 GPUs to train the model. If you use less GPUs, please change the learning rate and batch size in the config file.

To visualize the training proces:

```bash
visualdl --logdir vdl_log_dir
```

**step.4** run the following command to eval and Pedestrian Detection

```bash
python tools/eval.py -c configs/smalldet/ppyoloe_plus_sod_crn_l_40e_coco.yml -o weights=output/best_model/model.pdparams
```

To see the visualization results, you can run the following command and check the output folder.

```bash
python tools/vis_det.py --vis_task det --filter_score 0.3 
``` 

### TODO LIST 

- [x] Prepare Dataset
- [x] Train and Evaluate Model
- [x] Pedestrian Detection
- [ ] Pedestrian Tracking Prediction
- [ ] Pedestrian Tracking Prediction analysis
- [ ] Filter mistakes annotations


### Citation

```bash
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}
```