import argparse
import os
import cv2 as cv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='det', help="visualization task, choose in [gt, det]")
    args = parser.parse_args()
    return args

def main(args):
    data_root = './dataset/standford_campus'
    annotations_root = './dataset/standford_campus/image_annotations'
    videos_root = './dataset/standford_campus/videos'

    data_info = dict()
    scene_name = os.listdir(videos_root)
    for name in scene_name:
        data_info[name] = []

    # for name in scene_name:
    #     videos_path = os.path.join(videos_root, name)
    #     videos = os.listdir(videos_path)
    #     # single video path
    #     for num in videos:
    #         video_path = os.path.join(videos_path, num, 'video.mov')
    #         annotations_path = os.path.join(annotations_root, name, num, 'annotations.txt')
    #         raw = cv.VideoCapture(video_path)
    #         ret, frame = raw.read()
    #         frame_count = raw.get(cv.CAP_PROP_FRAME_COUNT)
    #         data_info[name].append({
    #             'annotations_path': annotations_path,
    #             'video_path': video_path,
    #             'video': num,
    #             'frame shape': frame.shape[:2],
    #             'frame_count': int(frame_count)
    #         })


if __name__ == "__main__":
    args = parse_args()
    main(args)