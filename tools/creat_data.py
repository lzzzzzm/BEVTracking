import cv2 as cv
import numpy as np
import json
import os
import shutil

import argparse
from pprint import pprint

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
        '--split_ratio', type=float, default=0.2, help="train and val split ratio")
    parser.add_argument(
        '--split_fps', type=int, default=10, help="the fps of clip video")
    parser.add_argument(
        '--crop_size', type=int, default=125, help="crop image size to filter bbox out of image")
    parser.add_argument(
        '--vis_create_data', action='store_true', help="visualize the create data")
    args = parser.parse_args()
    return args


def convert_video_to_image(video_path, image_path, anno_path, anno_info, split_fps, crop_size=50, vis_number=100):
    raw = cv.VideoCapture(video_path)
    index = 0
    write_index = 0
    while raw.isOpened():
        ret, frame = raw.read()
        if not ret:
            break
        if index % split_fps == 0:
            anno = anno_info[index]
            json_dict = []
            # crop frame
            frame = frame[crop_size:-crop_size, crop_size:-crop_size]
            for i in range(len(anno['bbox'])):
                w, h = anno['bbox'][i][2] - anno['bbox'][i][0], anno['bbox'][i][3] - anno['bbox'][i][1]
                bbox_xywh = [anno['bbox'][i][0], anno['bbox'][i][1], w, h]
                area = w * h
                json_dict.append({
                    'area': area,
                    'bbox': bbox_xywh,
                    'category_id': anno['category_id'][i],
                    'track_id': anno['track_id'][i],
                    'mask': anno['mask'][i],
                    'image_id': str(write_index).zfill(7),
                    'iscrowd': 0,
                    'id': i,
                })

            write_anno_path = os.path.join(anno_path, str(write_index).zfill(7) + '.json')
            write_image_path = os.path.join(image_path, str(write_index).zfill(7) + '.jpg')
            write_index = write_index + 1
            json.dump(json_dict, open(write_anno_path, 'w'))
            if args.vis_create_data:
                if write_index < vis_number:
                    img = draw_bbox(frame.copy(), anno['bbox'], anno['category_id'], anno['track_id'])
                    cv.imshow('img', img)
                    cv.waitKey(1)
                elif write_index == vis_number:
                    cv.destroyWindow('img')
            cv.imwrite(write_image_path, frame)

        index = index + 1


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def type_change(x):
    return int(float(x))


def read_annotations(annotations_path, max_frame, frame_shape, crop_size):
    with open(annotations_path) as f:
        anno = f.read().split('\n')
        max_track_id = int(anno[-2].split(' ')[0])

    frame_id_anno_info = dict()
    for frame_id in range(max_frame):
        frame_id_anno_info[frame_id] = {'bbox': [], 'category_id': [], 'track_id': [], 'mask': [], 'occulded': []}

    track_id_anno_info = dict()
    for track_id in range(max_track_id + 1):
        track_id_anno_info[track_id] = {'bbox': [], 'category_id': [], 'frame_id': []}

    # track id bool state
    track_id_keep = dict()
    for track_id in range(max_track_id + 1):
        track_id_keep[track_id] = {
            'track_state': False,
            'state_count': 0,
        }

    with open(annotations_path) as f:
        anno = f.read().split('\n')
        for line in anno:
            line = line.split(' ')
            if len(line) == 1:
                break
            frame_id = int(line[5])
            # bbox: xmin ymin xmax ymax
            bbox = [int(line[1]), int(line[2]), int(line[3]), int(line[4])]
            category_id = categories_map[line[9][1:-1]]
            track_id = int(line[0])
            occulded = int(line[7])
            # filter bbox out of image, center on the image
            if bbox[0] < crop_size or bbox[1] < crop_size or bbox[2] > frame_shape[1] - crop_size or bbox[3] > \
                    frame_shape[0] - crop_size:
                if track_id_keep[track_id]['track_state']:
                    track_id_keep[track_id]['state_count'] = track_id_keep[track_id]['state_count'] + 1
                track_id_keep[track_id]['track_state'] = False

            if bbox[0] > crop_size and bbox[1] > crop_size and bbox[2] < frame_shape[1] - crop_size and bbox[3] < \
                    frame_shape[0] - crop_size:
                if track_id_keep[track_id]['state_count'] == 0:
                    track_id_keep[track_id]['track_state'] = True

            if not track_id_keep[track_id]['track_state']:
                continue

            if category_id == 3 or category_id == 4 or category_id == 5:
                continue

            # crop bbox
            bbox = [bbox[0] - crop_size, bbox[1] - crop_size, bbox[2] - crop_size, bbox[3] - crop_size]

            track_id_anno_info[track_id]['bbox'].append(bbox)
            track_id_anno_info[track_id]['category_id'].append(0)
            track_id_anno_info[track_id]['frame_id'].append(frame_id)

            frame_id_anno_info[frame_id]['bbox'].append(bbox)
            frame_id_anno_info[frame_id]['category_id'].append(0)
            frame_id_anno_info[frame_id]['track_id'].append(track_id)
            frame_id_anno_info[frame_id]['occulded'].append(occulded)

    # track id bool state
    track_id_keep = dict()
    for track_id in range(max_track_id + 1):
        track_id_keep[track_id] = True

    # filter track id based on its static state
    for track_id in range(max_track_id + 1):
        state_count = 0
        bboxs = track_id_anno_info[track_id]['bbox']
        box_frame = len(bboxs)
        if box_frame == 0:
            continue
        x0min, y0min, x0max, y0max = bboxs[0]
        for bbox in bboxs[1:]:
            xmin, ymin, xmax, ymax = bbox
            if np.abs(x0min - xmin) == 0 and np.abs(y0min - ymin) == 0 \
                    and np.abs(x0max - xmax) == 0 and np.abs(y0max - ymax) == 0:
                state_count = state_count + 1
            x0min, y0min, x0max, y0max = xmin, ymin, xmax, ymax
            if state_count > box_frame * 0.98:
                track_id_keep[track_id] = False
                break

    for frame_id in range(max_frame):
        # delete the track id which is false
        track_id = frame_id_anno_info[frame_id]['track_id']
        for index, id in enumerate(track_id):
            if not track_id_keep[id]:
                frame_id_anno_info[frame_id]['mask'].append(False)
            else:
                frame_id_anno_info[frame_id]['mask'].append(True)
    return frame_id_anno_info


def draw_bbox(image, bboxs, category_ids, track_id, box_type='x1y1x2y2'):
    color_map = {
        0: (0, 255, 0),
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
        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
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
    image_path = os.path.join(data_root, 'images')
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
            anno_info = read_annotations(anno_path, info['frame_count'], info['frame shape'], args.crop_size)
            convert_video_to_image(info['video_path'], video_image_path, anno_image_path, anno_info, args.split_fps,
                                   args.crop_size)

    # covert to coco format ------------------------------------------------------
    coco_train_image_path = os.path.join(data_root, 'train')
    coco_val_image_path = os.path.join(data_root, 'val')
    check_dir(coco_train_image_path)
    check_dir(coco_val_image_path)

    train_coco_json_dict = {'annotations': [], 'images': [], 'categories': [{'id': 0, 'name': 'Pedestrian'}]}
    val_coco_json_dict = {'annotations': [], 'images': [], 'categories': [{'id': 0, 'name': 'Pedestrian'}]}

    print('convert annotations to coco format-----------------------')
    train_index = 0
    val_index = 0
    for scene in scene_name:
        videos_path = os.path.join(videos_root, scene)
        videos = os.listdir(videos_path)
        video_num = len(videos)
        train_video_num = int(video_num * (1 - args.split_ratio))
        index = 0
        for num in videos:
            if index < train_video_num:
                print('processing train scene:{}, video:{}'.format(scene, num))
            else:
                print('processing val scene:{}, video:{}'.format(scene, num))

            images_dir = os.path.join(image_path, scene, num)
            images_name = os.listdir(images_dir)
            img_path = os.path.join(images_dir, images_name[0])
            img_example = cv.imread(img_path)
            height, width = img_example.shape[:2]
            for img_name in images_name:
                img_path = os.path.join(images_dir, img_name)
                ann_path = os.path.join(image_annotations_path, scene, num, img_name[:-4] + '.json')
                anno_info = json.load(open(ann_path))

                if index < train_video_num:
                    train_name = '{}_{}_{}.jpg'.format(scene, num, train_index)
                    # rewrite the image_id
                    for i in range(len(anno_info)):
                        anno_info[i]['image_id'] = train_index
                        anno_info[i]['iscrowd'] = 0
                        anno_info[i]['id'] = len(train_coco_json_dict['annotations'])
                        train_coco_json_dict['annotations'].append(anno_info[i])
                    write_path = os.path.join(coco_train_image_path, train_name)
                    images_info = {
                        'file_name': train_name,
                        'height': height,
                        'width': width,
                        'id': train_index
                    }
                    train_index = train_index + 1
                    train_coco_json_dict['images'].append(images_info)
                else:
                    val_name = '{}_{}_{}.jpg'.format(scene, num, val_index)
                    # rewrite the image_id
                    for i in range(len(anno_info)):
                        anno_info[i]['image_id'] = val_index
                        anno_info[i]['iscrowd'] = 0
                        anno_info[i]['id'] = len(val_coco_json_dict['annotations'])
                        val_coco_json_dict['annotations'].append(anno_info[i])
                    write_path = os.path.join(coco_val_image_path, val_name)
                    images_info = {
                        'file_name': val_name,
                        'height': height,
                        'width': width,
                        'id': val_index
                    }
                    val_index = val_index + 1
                    val_coco_json_dict['images'].append(images_info)

                shutil.copy(img_path, write_path)
            index = index + 1

    # write coco json
    train_coco_json_path = os.path.join(annotations_root, 'train.json')
    val_coco_json_path = os.path.join(annotations_root, 'val.json')
    json.dump(train_coco_json_dict, open(train_coco_json_path, 'w'))
    json.dump(val_coco_json_dict, open(val_coco_json_path, 'w'))


if __name__ == "__main__":
    args = parse_args()
    main(args)
