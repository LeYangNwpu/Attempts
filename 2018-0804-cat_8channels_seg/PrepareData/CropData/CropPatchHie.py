from PIL import Image
import os
from multiprocessing import Pool

import CropPatch

hie_synmask_rp = '/home/yangle/TCyb/dataset/MSRA10K/val_formask/'
img_rp = '/home/yangle/TCyb/dataset/MSRA10K/val/'
gt_rp = '/home/yangle/TCyb/dataset/MSRA10K/valannot/'
hie_save_rp = '/home/yangle/TCyb/dataset/cat_128_del/ValPatch/'


def process(fol_name):
    save_rp = hie_save_rp + fol_name + '/'
    os.makedirs(save_rp)

    # context image
    raw_img_name = str(int(fol_name)) + '.png'
    cont = Image.open(img_rp + raw_img_name)
    cont = cont.resize((128, 128))
    res_img_name = fol_name + '.png'
    cont.save(save_rp + res_img_name)

    # crop patch: image, gt, synthesize mask
    synmask_rp = hie_synmask_rp + fol_name + '/'
    img = Image.open(img_rp + raw_img_name)
    gt = Image.open(gt_rp + raw_img_name)
    CropPatch.CropPatch(img, gt, synmask_rp, save_rp)


if __name__ == '__main__':
    num_processor = 10
    name_set = os.listdir(hie_synmask_rp)
    pool = Pool(num_processor)
    pool.map(process, name_set)
    pool.close()
    pool.join()
    print('process finished')

