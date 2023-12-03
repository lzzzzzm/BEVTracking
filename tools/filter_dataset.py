import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='det', help="visualization task, choose in [gt, det]")
    args = parser.parse_args()
    return args

def main(args):
    data_root = './dataset/standford_campus'
    annotations_root = './dataset/standford_campus/annotations'
    videos_root = './dataset/standford_campus/videos'

if __name__ == "__main__":
    args = parse_args()
    main(args)