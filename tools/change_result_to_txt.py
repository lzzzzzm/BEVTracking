import json
import os
import collections

json_path = 'out_vis/bbox.json'

json_data = json.load(open(json_path))
data1 = sorted(json_data, key=lambda x: x['file_name'], reverse=False)
wirte_list = []
for data in json_data:
    frame_id = data['image_id']+1
    category_id = data['category_id']
    bbox = data['bbox']
    x, y, w, h= bbox
    score = data['score']

    wirte_txt = '{},{},{},{},{},{},{}\n'.format(frame_id, int(x), int(y), int(w), int(h), score, 0)
    wirte_list.append(wirte_txt)

with open('out_vis/video0.txt', 'w') as f:
    for str in wirte_list:
        f.writelines(str)

