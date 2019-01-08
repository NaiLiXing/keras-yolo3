import logging
import argparse
from yolo import YOLO
from PIL import Image
import numpy as np
import os
import json
logging.basicConfig(filename='logger.txt', level=logging.INFO)
from pprint import pprint
root_path = '/home/xingnaili/Downloads/coco2017/2014/val2014/'
root_path2017 = '/home/xingnaili/Downloads/coco2017/train2017/'

clss_mapper = {}
gt_mapper = {}


def loadjson(filename):
    with open(filename) as json_file:
        person = json.load(json_file)
    return person


original_data = loadjson('./coco_data/instances_val2017.json')

annotations = original_data['annotations']
new_annotations = [ele for ele in annotations if ele['category_id'] == 1]

imgs_id_list = [ele['image_id'] for ele in new_annotations]

imgs_id_list = list(set(imgs_id_list))


train_data = loadjson('/home/xingnaili/Downloads/coco2017/annotations_train/instances_train2017.json')

ttannotations = train_data['annotations']
ttnew_annotations = [ele for ele in ttannotations if ele['category_id'] == 1]

ttimgs_id_list = [ele['image_id'] for ele in ttnew_annotations]

ttimgs_id_list = list(set(ttimgs_id_list))

images = train_data['images']

image_path_list = [ele['file_name'] for ele in images if ele['id'] in ttimgs_id_list]
image_path_list = list(set(image_path_list))
print(len(image_path_list))
with open('model_data/coco_classes.txt', 'r') as coco_file:
    for i in range(1, 81, 1):
        clss_mapper[coco_file.readline().strip()] = i

with open('model_data/gt_classes.txt', 'r') as gt_file:
    for i in range(1, 92, 1):
        gt_mapper[gt_file.readline().strip()] = i
pprint(gt_mapper)


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]  # bbox打分

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]
    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # print(ovr)
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]

    return keep


def detect_img(yolo):
    # images = os.listdir('/home/xingnaili/Downloads/coco2017/train2017')
    # images = os.listdir('/home/xingnaili/Downloads/coco2017/2014/val2014')
    # images = ['000000334417.jpg']
    resu = []
    for img in image_path_list:
        try:
            image = Image.open(root_path2017+img)
            result, image = yolo.detect_image(image)
            # image.show()
            if result:
                for ele in result:
                        inner = {"image_id": int(img[:-4]),
                                 "category_id": gt_mapper[ele['predicted_class']],
                                 "bbox": ele['bbox'],
                                 "score": ele['score'].tolist()}
                        if inner["image_id"] in imgs_id_list:
                            resu.append(inner)
        except Exception as E:
            logging.error(img)
            continue
    result = []

    ele_list = {}
    res_dict = {}
    for ele in resu:
        if ele['image_id'] not in res_dict:
            res_dict[ele['image_id']] = list()
        if ele['image_id'] not in ele_list:
            ele_list[ele['image_id']] = list()
        inner = [ele['bbox'][0], ele['bbox'][1], ele['bbox'][2] + ele['bbox'][0], ele['bbox'][3] + ele['bbox'][1]]
        inner.append(ele['score'])
        res_dict[ele['image_id']].append(inner)
        ele_list[ele['image_id']].append(ele)
    for image_id in res_dict:

        inner = res_dict[image_id]
        matrix_five = np.asarray(inner)
        keep_result = py_cpu_nms(matrix_five, 0.7)
        for i in keep_result:
            result.append(ele_list[image_id][i])

    # print(result)
    # print(type(result))
    # print(type(result[0]))
    # print(type(result[0]['image_id']))
    # print(type(result[0]['category_id']))
    # print(type(result[0]['bbox']))
    # print(type(result[0]['score']))
    # print(type(result[0]['bbox']))
    # print(type(result[0]['bbox'][0]))

    with open('./trained25987_w_person_2017_result_traindata.json', 'a') as json_file:
        json.dump(resu, json_file)
    yolo.close_session()


FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if True:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
