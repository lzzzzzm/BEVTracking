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
so we map the biker, skater to pedestrian, and filter other categories

### Installation

bevtracking requires the following dependencies (tested on Windows 10 and Ubuntu 18.04):

**step.1** install paddle
    
```bash
python -m pip install paddlepaddle-gpu==2.5.2.post116 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```
**step.2** install requirements

```bash
pip install -r requirements.txt
```

### Running

**step.1** download the dataset and put it in the data folder

**step.2** run the following command to prepare the dataset

```bash
python tools/create_data.py
```

**step.3** run the following command to train the model

```bash
python tools/train.py -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml
```

### Citation