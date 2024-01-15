import argparse
import os
import cv2 as cv

import json
from pycocotools.coco import COCO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vis_task', type=str, default='det', help="visualization task, choose in [gt, det]")
    parser.add_argument(
        '--vis_num', type=int, default=100, help="the number of visualization images, only work when vis_task is gt and det, -1 means all")
    parser.add_argument(
        '--pred_result', type=str, default='bbox.json', help="the det prediction result file path")
    parser.add_argument(
        '--filter_score', type=float, default=0.3, help="the score threshold for visualization")
    parser.add_argument(
        '--show_img', type=bool, default=False, help="whether to show image")
    parser.add_argument(
        '--write_img', type=bool, default=True, help="whether to wirte results")
    args = parser.parse_args()
    return args

def draw_bbox(img, bbox, score=1.0, color=(0, 255, 0), thickness=2):
    # coco format bbox
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    # draw score
    text_scale = max(1, img.shape[1] / 1600.)
    text_thickness = 2
    text = '{:.2f}'.format(float(score))
    cv.putText(
        img,
        text, (int(x1), int(y1) + 10),
        cv.FONT_HERSHEY_PLAIN,
        text_scale, (0, 255, 255),
        thickness=text_thickness)

    return img

def vis_gt(args):
    val_images_path = os.path.join('dataset', 'standford_campus', 'val')
    val_anno_path = os.path.join('dataset', 'standford_campus', 'annotations', 'val.json')

    coco = COCO(val_anno_path)
    ids = list(sorted(coco.imgs.keys()))
    for img_id in ids[:args.vis_num]:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        targets = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(val_images_path, path)
        img = cv.imread(img_path)
        for target in targets:
            bbox = target['bbox']
            img = draw_bbox(img, bbox)

        cv.imshow('img', img)
        cv.waitKey()

def vis_det(args):
    out_dir = 'output/det'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


    pred_result = json.load(open(args.pred_result))
    val_images_path = os.path.join('dataset', 'standford_campus', 'val')
    # sort pred_result by image_id
    pred_result = sorted(pred_result, key=lambda x: x['image_id'])
    val_anno_path = os.path.join('dataset', 'standford_campus', 'annotations', 'val.json')
    coco = COCO(val_anno_path)

    for img_id in range(args.vis_num):
        path = coco.loadImgs(img_id+1)[0]['file_name']
        img_path = os.path.join(val_images_path, path)
        img = cv.imread(img_path)
        # find img_id result
        img_result = list(filter(lambda x: x['image_id'] == img_id+1, pred_result))
        # filter score
        img_result = list(filter(lambda x: x['score'] > args.filter_score, img_result))
        for result in img_result:
            bbox = result['bbox']
            score = result['score']
            img = draw_bbox(img, bbox, score)

        if args.show_img:
            cv.imshow('img', img)
            cv.waitKey()

        if args.write_img:
            print('writing:', path)
            wirte_path = os.path.join(out_dir, path)
            cv.imwrite(wirte_path, img)



def main(args):
    if args.vis_task == 'gt':
        print('visualize ground truth')
        vis_gt(args)
    if args.vis_task == 'det':
        print('visualize detection result')
        vis_det(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
