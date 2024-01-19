import argparse
import os
import cv2 as cv

import json
from pycocotools.coco import COCO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir', type=str, default='out_mot/mot_outputs/video0', help="image_dir")
    args = parser.parse_args()
    return args


def main(args):
    image_dir = args.image_dir
    image_names = os.listdir(image_dir)
    image_names.sort()
    # get image width and height
    image_path = os.path.join(image_dir, image_names[0])
    img = cv.imread(image_path)
    height, width, _ = img.shape


    video_name = 'giv_12_pred_8.mp4'
    # set the format of video
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv.VideoWriter_fourcc(*'MJPG')
    # corrsponding the height and width
    videoWriter = cv.VideoWriter(video_name, fourcc, 10, (1024, 1024))
    index = 0
    for image_name in image_names:
        index = index +1
        image_path = os.path.join(image_dir, image_name)
        img = cv.imread(image_path)
        img = cv.resize(img, (1024, 1024))
        videoWriter.write(img)
        if index > 200:
            break
    videoWriter.release()


if __name__ == '__main__':
    args = parse_args()
    main(args)
