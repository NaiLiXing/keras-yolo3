import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import json
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = 'bbox'

#
annFile = './coco_data/person_instances_val2017.json'
cocoGt = COCO(annFile)


resFile = './trained25987_w_person_2017_result.json'
cocoDt = cocoGt.loadRes(resFile)


# annFile = './coco_data/instances_val2014.json'
# cocoGt = COCO(annFile)
#
#
# resFile = './coco_data/instances_val2014_fakebbox100_results.json'
# cocoDt = cocoGt.loadRes(resFile)

# imgIds = sorted(cocoGt.getImgIds())
# resFile = './corr_wh_pertrain_person_2017.json'
# anns = json.load(open(resFile))
# imgIds = list(set([ele['image_id'] for ele in anns]))

cocoEval = COCOeval(cocoGt, cocoDt, annType)
# cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
