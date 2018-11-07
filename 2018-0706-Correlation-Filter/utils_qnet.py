# coding: utf-8
import torch
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision

import tiramisu

def ini_seg_model(fpath=None):
    if fpath is None:
        fpath = '/home/yangle/code/CorFilter/ext_fea/tiramisu12/weights/latest_weights.pth'
    seg_model = tiramisu.FCDenseNet57(n_classes=2)
    seg_model = seg_model.cuda()
    seg_model.eval()
    print('load the segmentation model')
    state = torch.load(fpath)
    seg_model.load_state_dict(state['state_dict'])
    return seg_model

seg_model = ini_seg_model()


def np_to_pil_image(npimg, mode='F'):
    np_img = np.transpose(npimg, (1, 0))
    pil_img = Image.fromarray(np_img, mode=mode)
    return pil_img

def pil_image_to_np(pil_img):
    np_img = np.asarray(pil_img)
    npimg = np.transpose(np_img, (1, 0))
    return npimg


def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
	"""Sets the learning rate to the initially
		configured `lr` decayed by `decay` every `n_epochs`"""
	new_lr = lr * (decay ** (cur_epoch // n_epochs))
	for param_group in optimizer.param_groups:
		param_group['lr'] = new_lr

def get_enl_box(box, fac, hei, wid):
    # print(type(box))
    c_col = box[0]
    c_row = box[1]
    width = box[2]
    height = box[3]
    width_half = width / 2
    height_half = height / 2
    col_min_tmp = c_col - width_half
    col_max_tmp = c_col + width_half
    row_min_tmp = c_row - height_half
    row_max_tmp = c_row + height_half

    # value for emergency return
    col_min_re = int(col_min_tmp)
    col_max_re = int(col_max_tmp)
    row_min_re = int(row_min_tmp)
    row_max_re = int(row_max_tmp)

    col_min = col_min_tmp * (1 - fac)
    col_min = int(col_min)
    if col_min < 0:
        col_min = 0
    # col_max_tmp = c_col + width_half
    col_max = col_max_tmp + (wid - col_max_tmp) * fac
    col_max = int(col_max)
    if col_max > wid - 1:
        col_max = wid - 1
    if col_min >= col_max:
        # print(col_min, col_max, row_min, row_max)
        print('the feature map is too tiny, emergency return')
        # print(col_min_re, col_max_re, row_min_re, row_max_re)
        return col_min_re, col_max_re, row_min_re, row_max_re

    # row_min_tmp = c_row - height_half
    row_min = row_min_tmp * (1 - fac)
    row_min = int(row_min)
    if row_min < 0:
        row_min = 0
    # row_max_tmp = c_row + height_half
    row_max = row_max_tmp + (hei - row_max_tmp) * fac
    row_max = int(row_max)
    if row_max > hei - 1:
        row_max = hei - 1
    if row_min >= row_max:
        print('the feature map is too tiny, emergency return')
        return col_min_re, col_max_re, row_min_re, row_max_re
        # print(col_min, col_max, row_min, row_max)
    # print('original box')
    # print(col_min_tmp, col_max_tmp, row_min_tmp, row_max_tmp)
    # print('enlarged box')
    # print(col_min, col_max, row_min, row_max)
    return col_min, col_max, row_min, row_max

def resize_img(img_tmp, cha, hei, wid):
    # mode = 'F'
    img_tmp_np = np.zeros((cha, hei, wid))
    for icha in range(cha):
        img_ch = img_tmp[icha, :, :]
        img_ch_tif = np_to_pil_image(img_ch)
        img_ch_std = img_ch_tif.resize((hei, wid))
        img_tmp_np[icha, :, :] = pil_image_to_np(img_ch_std)
    return img_tmp_np


def resize_mask(mask_np, col_min, col_max, row_min, row_max):
    # print('mask_np.shape')
    # print(mask_np.shape)
    cha, hei, wid = mask_np.shape
    # print('col_max-col_min, row_max-row_min')
    # print(col_max-col_min, row_max-row_min)
    mask_tmp_np = np.zeros((cha, col_max-col_min, row_max-row_min))
    for icha in range(cha):
        mask_ch = mask_np[icha,:,:]
        mask_ch_tif = np_to_pil_image(mask_ch)
        mask_ch_std = mask_ch_tif.resize((col_max-col_min, row_max-row_min))
        # print('mask_ch_std.size')
        # print(mask_ch_std.size)
        mask_tmp_np[icha, :, :] = pil_image_to_np(mask_ch_std)
    return mask_tmp_np


def resize_target(target_tmp, hei, wid):
    # mode = 'F'
    target_tmp_tif = np_to_pil_image(target_tmp)
    target_tmp_std = target_tmp_tif.resize((hei, wid))
    return pil_image_to_np(target_tmp_std)


# def crop_img(imgBat, boxBat, facBat):
#     bat, cha, hei, wid = imgBat.size()
#     # print('bat, cha, hei, wid')
#     # print(bat, cha, hei, wid)
#     boxBat = torch.round(boxBat * hei)
#     boxBat = boxBat.numpy()
#     facBat = facBat.cpu().data
#     facBat = facBat.numpy()
#     imgBat_np = np.zeros((bat, cha, hei, wid))
#     for ibat in range(bat):
#         ###############
#         # enlarge box #
#         ###############
#         box = boxBat[ibat, :]
#         fac = facBat[ibat]
#         col_min, col_max, row_min, row_max = enlarge_box(box, fac, hei, wid)
#
#         #############################
#         # crop and resize the image #
#         #############################
#         # print('crop and resize the image')
#         img_ori = imgBat[ibat, :]
#         # print(col_min, col_max, row_min, row_max)
#         img_crop = img_ori[:, col_min:col_max, row_min:row_max]
#         img_tmp = img_crop.numpy()
#         imgBat_np[ibat, :, :, :] = resize_img(img_tmp, cha, hei, wid)
#
#         # # the target is hei * wid
#         # target_ori = targetsBat[ibat, :]
#         # target_crop = target_ori[col_min:col_max, row_min:row_max]
#         # target_tmp = target_crop.numpy()
#         # targetsBat_np[ibat, :, :] = resize_target(target_tmp, hei, wid)
#
#     img_ret = torch.from_numpy(imgBat_np)
#     img_ret = img_ret.type(torch.FloatTensor)
#     # tar_ret = torch.from_numpy(targetsBat_np)
#     # tar_ret = tar_ret.type(torch.LongTensor)
#
#     return img_ret


def get_ori_box(box):
    c_col = box[0]
    c_row = box[1]
    width = box[2]
    height = box[3]
    width_half = width / 2
    height_half = height / 2
    col_min = int(c_col - width_half)
    col_max = int(c_col + width_half)
    row_min = int(c_row - height_half)
    row_max = int(c_row + height_half)

    return col_min, col_max, row_min, row_max


def get_seg_masks(imgBat, boxBat, facBat):
    bat, cha, hei, wid = imgBat.size()
    #convert data
    boxBat = torch.round(boxBat * hei)
    boxBat = boxBat.numpy()
    facBat = facBat.cpu().data
    facBat = facBat.numpy()
    #original segmentation
    seg_input_ori = Variable(imgBat.cuda())
    seg_output_ori = seg_model(seg_input_ori)
    # seg_output_ori = seg_output_ori.cpu().data
    seg_output_crop = seg_output_ori.cpu().data.numpy()

    for ibat in range(bat):
        img_t = imgBat[ibat,:,:,:]
        box_data = boxBat[ibat,:]
        fac = facBat[ibat]
        # #original box data
        # col_min_ori, col_max_ori, row_min_ori, row_max_ori = get_ori_box(box_data)
        #enlarged box data
        col_min, col_max, row_min, row_max = get_enl_box(box_data, fac, hei, wid)
        #crop image
        img_crop = img_t[:, col_min:col_max, row_min:row_max]
        img_crop = img_crop.numpy()
        #obtain segmentation mask
        img_np = resize_img(img_crop, cha, hei, wid)
        img_t = torch.from_numpy(img_np)
        img_t = img_t.type(torch.FloatTensor)
        img_t = img_t.view(1, cha, hei, wid)
        img_t = Variable(img_t.cuda())
        #segmentation 224*224
        mask_t = seg_model(img_t)
        mask_t = mask_t.view(2, hei, wid)
        #resize mask
        mask_np = mask_t.cpu().data.numpy()
        mask_np_tmp = resize_mask(mask_np, col_min, col_max, row_min, row_max)
        # print('mask_np_tmp.shape')
        # print(mask_np_tmp.shape)
        #replace the mask
        seg_output_crop[ibat,:,col_min:col_max, row_min:row_max] = mask_np_tmp
    mask_ret = torch.from_numpy(seg_output_crop)
    mask_ret = mask_ret.type(torch.FloatTensor)
    maskBat = Variable(mask_ret.cuda())

    return seg_output_ori, maskBat


def error(preds, targets):
	assert preds.size() == targets.size()
	bs, h, w = preds.size()
	n_pixels = bs * h * w
	incorrect = preds.ne(targets).cpu().sum()
	err = 100. * incorrect / n_pixels
	return round(err, 5)


def get_predictions(output_batch):
	# Variables(Tensors) of size (bs,12,224,224)
	bs, c, h, w = output_batch.size()
	tensor = output_batch.data
	# Argmax along channel axis (softmax probabilities)
	values, indices = tensor.cpu().max(1)
	indices = indices.view(bs, h, w)

	return indices

def train(model, trn_loader, optimizer):
    model.train()
    trn_loss = 0
    trn_error = 0


    for imgs, targets, fea_np, box, nlist in trn_loader:

        # prepare feature data #
        fea = fea_np.type(torch.FloatTensor)
        bat, length = fea.size()
        fea_tmp = fea.view(bat, 1, 1, length)
        fea_tmp = fea_tmp.cuda()
        fea = Variable(fea_tmp)

        # calculate Q-net output #
        output_data = model(fea)
        fac = output_data
        # print(fac.cpu().data)

        # calculate segmentation mask
        #seg_output_ori, maskBat
        seg_output_ori, seg_output = get_seg_masks(imgs, box, fac)

        # segmentation loss
        seg_criterion = nn.NLLLoss2d().cuda()
        # targets = Variable(targets_c.cuda())
        targets = Variable(targets.cuda())
        seg_loss = seg_criterion(seg_output, targets)

        #visualize seg_output and targets
        save_dir = '/home/yangle/dataset/ana/'
        bs, c, h, w = seg_output.size()
        tensor = seg_output.cpu().data
        value, index = tensor.max(1)
        index = index.view(bs, h, w)
        index_show = index.type(torch.FloatTensor)
        index_np = index_show.numpy()
        tar_show = targets.type(torch.FloatTensor)
        tar_np = tar_show.cpu().data.numpy()
        for ibats in range(bs):
            index_tmp_np = index_np[ibats, :, :]
            index_img = Image.fromarray(index_tmp_np)
            save_name_index = nlist[ibats]
            save_name_index = save_name_index[1:-4] + '_seg.tif'
            index_img.save(save_dir + save_name_index)
            #tar
            tar_tmp_np = tar_np[ibats, :, :]
            tar_img = Image.fromarray(tar_tmp_np)
            save_name_tar = nlist[ibats]
            save_name_tar = save_name_tar[1:-4] + '_tar.tif'
            tar_img.save(save_dir + save_name_tar)

        # optimizer.zero_grad()
        # output_data.backward(seg_loss)
        # optimizer.step()

        trn_loss += seg_loss.data[0]
        pred = get_predictions(seg_output)
        trn_error += error(pred, targets.data.cpu())


    trn_loss /= len(trn_loader)  # n_batches
    trn_error /= len(trn_loader)
    return trn_loss, trn_error


def test(model, test_loader):
    model.eval()
    test_loss = 0
    test_error = 0
    # resume pretrained segmentation model #
    # seg_model = ini_seg_model()

    for imgs, targets, fea_np, box, nlist in test_loader:
        # prepare feature data #
        fea = fea_np.type(torch.FloatTensor)
        bat, length = fea.size()
        fea_tmp = fea.view(bat, 1, 1, length)
        fea_tmp = fea_tmp.cuda()
        fea = Variable(fea_tmp)

        # calculate Q-net output #
        output_data = model(fea)
        fac = output_data

        seg_output_ori, seg_output = get_seg_masks(imgs, box, fac)

        # # crop image #
        # img_c, targets_c = crop_img(imgs, targets, box, fac, nlist)
        #
        #
        # # get segmentation mask #
        # # seg_input = Variable(img_c.cuda())
        # seg_input = Variable(imgs.cuda())
        # seg_output = seg_model(seg_input)

        # segmentation loss
        seg_criterion = nn.NLLLoss2d().cuda()
        # targets = Variable(targets_c.cuda())
        targets = Variable(targets.cuda())
        seg_loss = seg_criterion(seg_output, targets)

        test_loss += seg_loss.data[0]
        pred = get_predictions(seg_output)
        test_error += error(pred, targets.data.cpu())

    test_loss /= len(test_loader)  # n_batches
    test_error /= len(test_loader)
    return test_loss, test_error

