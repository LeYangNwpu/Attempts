# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import argparse
from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd
import pickle
from PIL import Image

import tiramisu_rms as tiramisu
import saliency_qnet as saliency
import experiment
import utils_qnet as utils
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--EXP_NAME', type=str, default='tiramisu12')
parser.add_argument('--EXP_DIR', type=str, default='/home/yangle/code/CorFilter/ext_fea/')
parser.add_argument('--DATASET_PATH', type=str, default='/home/yangle/dataset/QNet_hard/')
parser.add_argument('--SAVE_DIR', type=str, default='/home/yangle/dataset/QNet_hard/trainfea/')
parser.add_argument('--LEARNING_RATE', type=float, default=1e-4)
parser.add_argument('--WEIGHT_DECAY', type=float, default=0.0001)
args = parser.parse_args()


model = tiramisu.FCDenseNet57(n_classes=2)
# print(model)
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY)
exper = experiment.Experiment(args.EXP_NAME, args.EXP_DIR)
exper.resume(model, optimizer)

normalize = transforms.Normalize(mean=saliency.mean, std=saliency.std)
test_dset = saliency.TestImage(
	args.DATASET_PATH, 'train', joint_transform=None,
	transform=transforms.Compose([
		transforms.Scale((224, 224)),
		transforms.ToTensor(),
		normalize
	]))
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False)

count = 0
m = nn.Softmax()
m2d = nn.Softmax2d()
mSig = nn.Sigmoid()
for data, name_ori, box_ori in test_loader:
	data = Variable(data.cuda(), volatile=True)
	fea_list = model(data)
	box_c = box_ori[0, :]

	map_fea_np = np.zeros((16, 16, 6))
	# print('map_fea_np')
	# print(type(map_fea_np))

	for ifea in range(len(fea_list)):
		fea = fea_list[ifea]
		(bat, cha, hei, wid) = fea.size()
		box = torch.round(box_c * hei)
		c_col = int(box[0])
		c_row = int(box[1])
		width = int(box[2])
		width_half = width / 2
		height = int(box[3])
		height_half = height / 2
		col_min = c_col - width_half
		if col_min < 0:
			col_min = 0
		col_max = c_col + width_half
		if col_max > wid - 1:
			col_max = wid - 1
		if col_min >= col_max:
			print(col_min, col_max, row_min, row_max)
			print('the feature map is too tiny')
			continue

		row_min = c_row - height_half
		if row_min < 0:
			row_min = 0
		row_max = c_row + height_half
		if row_max > hei - 1:
			row_max = hei - 1
		if row_min >= row_max:
			print(col_min, col_max, row_min, row_max)
			print('the feature map is too tiny')
			continue

		# max pool via channel
		bs, c, h, w = fea.size()
		fea, fea_ind = torch.max(fea, 1)
		fea = fea.view(1, 1, h, w)

		# fea = m2d(fea)

		filmap = fea[:, :, col_min:col_max, row_min:row_max]
		filmap = filmap.contiguous()
		filmap = filmap.cuda()

		### correlation filter ###
		score_map = F.conv2d(fea, filmap)
		score_map = score_map.squeeze()
		# print('score map')
		# print(score_map)

		# # # softmax normlization
		# [num_col, num_row] = score_map.size()
		# score_map_tmp = score_map.view(1, num_col*num_row)
		# #####
		# # Q # the softmax is so strong that other location is zero, loss information
		# #####
		# score_map_tmp = m(score_map_tmp)
		# score_map = score_map_tmp.view(num_col, num_row)
		# print(score_map)

		# # sigmoid normalization
		# score_map = mSig(score_map)
		# print(score_map)

		#####
		# R # the proper normalization way: div the maximum value
		#####
		score_map = score_map / score_map.max()
		# print(score_map)
		# print(score_map.max())
		# print(score_map.min())
		# print(score_map.mean())

		### resize score map, concatnate them###
		c_map = score_map.cpu().data.numpy()
		mode = 'F'
		# map_img = Image.fromarray(c_map, mode=mode)
		map_img = utils.np_to_pil_image(c_map)
		map_std = map_img.resize((16, 16))
		# map_np = np.asarray(map_std)
		map_np = utils.pil_image_to_np(map_std)
		map_fea_np[:, :, ifea] = map_np
		# print('feature map')
		# print(map_np)

	map_global = map_fea_np.mean(axis=2)
	print(map_global)
	# print(type(map_global))
	# print(map_global.shape)
	fea_vec = np.reshape(map_global, 16*16)
	# print(fea_vec.shape)

	name = name_ori[0]
	name = name[0:-3]
	out_file = args.SAVE_DIR + name + 'pkl'
	output = open(out_file, 'wb')
	pickle.dump(fea_vec, output)
	output.close()


