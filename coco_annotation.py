import json
from collections import defaultdict

val_ana = '/home/xingnaili/Downloads/coco2017/annotations_train/instances_val2017.json'
train_ana = "/home/xingnaili/Downloads/coco2017/annotations_train/instances_train2017.json"
name_box_id = defaultdict(list)
id_name = dict()
f = open(
    "/home/xingnaili/Downloads/coco2017/annotations_train/instances_train2017.json",
    encoding='utf-8')
data = json.load(f)

annotations = data['annotations']
for ant in annotations:
    id = ant['image_id']
    name = '/home/xingnaili/Downloads/coco2017/train2017/%012d.jpg' % id
    cat = ant['category_id']

    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11

    name_box_id[name].append([ant['bbox'], cat])
print(len(name_box_id.keys()))
f = open('train_person.txt', 'w')
for key in name_box_id.keys():

    res = []
    box_infos = name_box_id[key]
    for info in box_infos:
        if info[1] == 0:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            res.append(box_info)
    if res:
        f.write(key)
        for box_info in res:
            f.write(box_info)
        f.write('\n')
f.close()
