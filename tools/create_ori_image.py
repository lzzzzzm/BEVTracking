import cv2 as cv
import numpy as np
import json
import os
import shutil

import argparse
from pprint import pprint
from PIL import Image, ImageDraw, ImageFont
from ppdet.utils.download import get_path

"""
    annotations format:
    Track ID, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label
    label categories ID map：
    Biker：0
    Pedestrian：1
    Skater：2
    Cart：3
    Car：4
    Bus：5

    in this project, we only evaluate the performance of pedestrian detection
    so we map the biker, skater to pedestrian, and filter other categories
"""

categories_map = {
    'Biker': 0,
    "Pedestrian": 1,
    'Skater': 2,
    'Cart': 3,
    'Car': 4,
    'Bus': 5
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--split_ratio', type=float, default=0.33, help="train and val split ratio")
    parser.add_argument(
        '--split_fps', type=int, default=10, help="the fps of clip video")
    parser.add_argument(
        '--crop_size', type=int, default=125, help="crop image size to filter bbox out of image")
    parser.add_argument(
        '--vis_create_data', action='store_true', help="visualize the create data")
    args = parser.parse_args()
    return args


def convert_video_to_image(video_path, image_path, anno_path, split_fps, crop_size=50, vis_number=100):
    raw = cv.VideoCapture(video_path)
    index = 0
    write_index = 0
    while raw.isOpened():
        ret, frame = raw.read()
        if not ret:
            break
        if index % split_fps == 0:
            write_image_path = os.path.join(image_path, str(write_index).zfill(6)+'.jpg')
            cv.imwrite(write_image_path, frame.copy())
            write_index = write_index + 1
        index = index + 1


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def type_change(x):
    return int(float(x))


def draw_bbox(image, bboxs, category_ids, track_id, box_type='x1y1x2y2'):
    color_map = {
        0: (0, 0, 0),
        1: (0, 0, 255),
        2: (255, 0, 0),
        3: (255, 255, 0),
        4: (0, 255, 255),
        5: (255, 0, 255)
    }
    for i in range(len(bboxs)):
        bbox = bboxs[i]
        if box_type == 'x1y1x2y2':
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        elif box_type == 'xywh':
            x1, y1, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            x2, y2 = x1 + w, y1 + h
        category_id = category_ids[i]
        color = color_map[category_id]
        cv.rectangle(image, (x1, y1), (x2, y2), color, 1)
        # write id on bbox lefttop
        cv.putText(image, str(track_id[i]), (bbox[0] + 5, bbox[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color)
    return image


def main(args):
    data_root = './dataset/standford_campus'
    annotations_root = './dataset/standford_campus/annotations'
    videos_root = './dataset/standford_campus/videos'

    data_info = dict()

    scene_name = os.listdir(videos_root)
    for name in scene_name:
        data_info[name] = []
    # read dataset info------------------------------------------------------
    for name in scene_name:
        videos_path = os.path.join(videos_root, name)
        videos = os.listdir(videos_path)
        # single video path
        for num in videos:
            video_path = os.path.join(videos_path, num, 'video.mov')
            annotations_path = os.path.join(annotations_root, name, num, 'annotations.txt')
            raw = cv.VideoCapture(video_path)
            ret, frame = raw.read()
            frame_count = raw.get(cv.CAP_PROP_FRAME_COUNT)
            data_info[name].append({
                'annotations_path': annotations_path,
                'video_path': video_path,
                'video': num,
                'frame shape': frame.shape[:2],
                'frame_count': int(frame_count)
            })
    image_path = os.path.join(data_root, 'ori_images')
    image_annotations_path = os.path.join(data_root, 'image_annotations')

    check_dir(image_path)
    check_dir(image_annotations_path)

    pprint(data_info)
    # convert video to image and write anno info------------------------------------------------------
    for scene in scene_name:
        scene_image_path = os.path.join(image_path, scene)
        scene_anno_path = os.path.join(image_annotations_path, scene)
        check_dir(scene_image_path)
        check_dir(scene_anno_path)
        for info in data_info[scene]:
            video_image_path = os.path.join(scene_image_path, info['video'])
            anno_image_path = os.path.join(scene_anno_path, info['video'])
            check_dir(video_image_path)
            check_dir(anno_image_path)
            anno_path = info['annotations_path']
            print('processing scene:{}, video:{}'.format(scene, info['video']))
            convert_video_to_image(info['video_path'], video_image_path, anno_image_path, args.split_fps,
                                   args.crop_size)



if __name__ == "__main__":
    args = parse_args()
    main(args)
