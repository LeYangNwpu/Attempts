import os
from PIL import Image
import numpy as np
import pickle

import jaccard
import f_boundary

gt_rp = '/disk1/hpl/segmentation/dataset/VAlannot/'
mask_rp = '/disk1/hpl/segmentation/results/results_resnet34_all/'

gt_set = os.listdir(gt_rp)
gt_set.sort()

mask_num = len(gt_set)
measure_j = np.zeros(mask_num + 1)
measure_f = np.zeros(mask_num + 1)

for iimg, gt_name in enumerate(gt_set):
    print(gt_name)
    gt = Image.open(gt_rp + gt_name)
    mask = Image.open(mask_rp + gt_name)
    mask = mask.resize(gt.size)
    mask = mask.convert('L')

    gt_np = np.asarray(gt, dtype=np.uint8)
    mask_np = np.asarray(mask, dtype=np.uint8)
    measure_j[iimg] = jaccard.db_eval_iou(gt_np, mask_np)
    measure_f[iimg] = f_boundary.db_eval_boundary(mask_np, gt_np)

mean_j = np.mean(measure_j)
measure_j[mask_num] = mean_j
mean_f = np.mean(measure_f)
measure_f[mask_num] = mean_f
print('measure jaccard: %f' % mean_j)
print('measure f: %f' % mean_f)
out = open('measure_jf_fc-resnet50-prelu.pkl', 'wb')
pickle.dump([measure_j, measure_f], out)
out.close()
