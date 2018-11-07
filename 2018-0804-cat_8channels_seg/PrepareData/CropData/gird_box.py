from PIL import Image
import numpy as np
import os

mask_dir = '/home/yangle/TCyb/dataset/tiramisu_gird_large/valori/'
res_dir = '/home/yangle/TCyb/dataset/tiramisu_gird_large/valbox/'

num_div = 8
edge_img = 128
edge_crop = edge_img / num_div
box_object = np.ones((edge_crop, edge_crop), dtype=np.uint8)
box_context = np.zeros((edge_crop, edge_crop), dtype=np.uint8)
num_patch_pix = edge_crop * edge_crop

mask_set = os.listdir(mask_dir)
for iimg in range(len(mask_set)):
# for iimg in range(2):
    name_mask = mask_set[iimg]

    img_path = mask_dir + name_mask
    box_path = res_dir + name_mask

    img = Image.open(img_path)
    img = np.asarray(img)
    box = np.zeros((edge_img, edge_img), dtype=np.uint8)

    for ihei in range(num_div):
        hei_min = ihei * edge_crop
        hei_max = (ihei+1) * edge_crop
        for iwid in range(num_div):
            wid_min = iwid * edge_crop
            wid_max = (iwid+1) * edge_crop
            patch = img[hei_min:hei_max, wid_min:wid_max]
            sum_pix = np.sum(patch)
            value_fill = 255.0 * sum_pix / num_patch_pix
            # value_fill = value_fill.astype(dtype=np.uint8)
            box[hei_min:hei_max, wid_min:wid_max] = value_fill

    box = Image.fromarray(box, mode='L')
    box.save(box_path)
