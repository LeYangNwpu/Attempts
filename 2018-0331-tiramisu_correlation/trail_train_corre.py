# coding: utf-8

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import os
import shutil

import dataset_7ch as saliency
import joint_transforms
import tiramisu as tiramisu
import experiment
import utils


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--DATASET_PATH', type=str, default='/home/yangle/TCyb/dataset/tiramisu_7ch_corre/')
	parser.add_argument('--RESULTS_PATH', type=str, default='/home/yangle/TCyb/result/TrainNet/results/')
	parser.add_argument('--WEIGHTS_PATH', type=str, default='/home/yangle/TCyb/result/TrainNet/models/')
	parser.add_argument('--EXPERIMENT', type=str, default='/home/yangle/TCyb/result/TrainNet/')
	parser.add_argument('--EXPNAME', type=str, default='EncDec_7ch_corre')
	parser.add_argument('--N_EPOCHS', type=int, default=1000)
	parser.add_argument('--MAX_PATIENCE', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--N_CLASSES', type=int, default=2)
	parser.add_argument('--LEARNING_RATE', type=float, default=1e-3)
	parser.add_argument('--LR_DECAY', type=float, default=0.9)
	parser.add_argument('--DECAY_LR_EVERY_N_EPOCHS', type=int, default=1)
	parser.add_argument('--WEIGHT_DECAY', type=float, default=0.0001)
	parser.add_argument('--CUDNN', type=bool, default=True)
	args = parser.parse_args()

	torch.cuda.manual_seed(args.seed)
	cudnn.benchmark = args.CUDNN

	normalize = transforms.Normalize(mean=saliency.mean, std=saliency.std)
	train_joint_transformer = transforms.Compose([joint_transforms.JointRandomHorizontalFlip()])

	train_dset = saliency.Saliency(
		args.DATASET_PATH, 'train', joint_transform=train_joint_transformer,
		transform=transforms.Compose([transforms.ToTensor(), normalize, ]))
	train_loader = torch.utils.data.DataLoader(
		train_dset, batch_size=args.batch_size, shuffle=True)

	val_dset = saliency.Saliency(
		args.DATASET_PATH, 'val',
		transform=transforms.Compose([transforms.ToTensor(), normalize, ]))
	val_loader = torch.utils.data.DataLoader(
		val_dset, batch_size=args.batch_size, shuffle=False)

	print("TrainImages: %d" % len(train_loader.dataset.imgs))
	print("ValImages: %d" % len(val_loader.dataset.imgs))

	# example_inputs, example_targets, _, _ = next(iter(train_loader))
	img, _, cont, _ = next(iter(train_loader))
	cont = Variable(cont)
	img = Variable(img)
	print('cont.size()')
	print(cont.size())
	print('img.size()')
	print(img.size())
	# correlation filter
	score_map = F.conv2d(cont, img)

	# linear normalize the score map
	score_map_norm = (score_map - score_map.min()) / (score_map.max() - score_map.min())
	score_map = score_map_norm.data
	# score_map_e = score_map.expand(-1, 2, 64, 64)
	print(score_map.size())
	# print(score_map_e.size())

	# print(type(score_map))
	# print(score_map.size())
	# print(score_map.min())
	# print(score_map.max())
	# score_map = score_map_norm.numpy()
	# map_show = score_map[0,0,:,:]
	# print(type(score_map))
	# print(score_map.shape)




if __name__=='__main__':
    main()

