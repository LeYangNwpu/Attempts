# coding: utf-8

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import time
import os
import shutil

import saliency_dataset as saliency
import joint_transforms
import tiramisu_preTrain as tiramisu_pre
import tiramisu as tiramisu
import experiment
import utils

# import torch.optim.lr_scheduler as lr_scheduler


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--DATASET_PATH', type=str, default='/home/yangle/dataset/DUTS/')
	parser.add_argument('--RESULTS_PATH', type=str, default='/home/yangle/result/TrainNet/results/')
	parser.add_argument('--WEIGHTS_PATH', type=str, default='/home/yangle/result/TrainNet/models/')
	parser.add_argument('--EXPERIMENT', type=str, default='/home/yangle/result/TrainNet/')
	parser.add_argument('--N_EPOCHS', type=int, default=300)
	parser.add_argument('--MAX_PATIENCE', type=int, default=30)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--N_CLASSES', type=int, default=2)
	parser.add_argument('--LEARNING_RATE', type=float, default=1e-4)
	parser.add_argument('--LR_DECAY', type=float, default=0.995)
	parser.add_argument('--DECAY_LR_EVERY_N_EPOCHS', type=int, default=1)
	parser.add_argument('--WEIGHT_DECAY', type=float, default=0.0001)
	parser.add_argument('--CUDNN', type=bool, default=True)
	args = parser.parse_args()

	torch.cuda.manual_seed(args.seed)
	cudnn.benchmark = args.CUDNN

	normalize = transforms.Normalize(mean=saliency.mean, std=saliency.std)
	train_joint_transformer = transforms.Compose([
		joint_transforms.JointResize(224),
		joint_transforms.JointRandomHorizontalFlip()
	])

	train_dset = saliency.Saliency(
		args.DATASET_PATH, 'train', joint_transform=train_joint_transformer,
		transform=transforms.Compose([transforms.ToTensor(), normalize, ]))
	train_loader = torch.utils.data.DataLoader(
		train_dset, batch_size=args.batch_size, shuffle=False)

	val_joint_transformer = transforms.Compose([joint_transforms.JointResize(224)])
	val_dset = saliency.Saliency(
		args.DATASET_PATH, 'val', joint_transform=val_joint_transformer,
		transform=transforms.Compose([transforms.ToTensor(), normalize, ]))
	val_loader = torch.utils.data.DataLoader(
		val_dset, batch_size=8, shuffle=False)

	print("TrainImages: %d" % len(train_loader.dataset.imgs))
	print("ValImages: %d" % len(val_loader.dataset.imgs))
	# print("TestImages: %d" % len(test_loader.dataset.imgs))

	example_inputs, example_targets = next(iter(train_loader))
	print("InputsBatchSize: ", example_inputs.size())
	print("TargetsBatchSize: ", example_targets.size())
	print("\nInput (size, max, min) ---")
	#input
	i = example_inputs[0]
	print(i.size())
	print(i.max())
	print(i.min())
	print("Target (size, max, min) ---")
	#target
	t = example_targets[0]
	print(t.size())
	print(t.max())
	print(t.min())

	######################################
	# load weights from pretrained model #
	######################################

	model_pre = tiramisu_pre.FCDenseNet57(in_channels=3, n_classes=2)
	model_pre = torch.nn.DataParallel(model_pre).cuda()
	fpath = '/home/yangle/result/TrainNet/segment/weights/segment-weights-132-0.109-4.278-0.120-4.493.pth'
	state = torch.load(fpath)
	pretrained_dict = state['state_dict']

	model = tiramisu.FCDenseNet57(in_channels=3, n_classes=2)
	model = torch.nn.DataParallel(model).cuda()
	model_dict = model.state_dict()

	# 1. filter out unnecessary keys
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict)
	# 3. load the new state dict
	model.load_state_dict(model_dict)
	# convert model trained with multiple GPUs into model within single GPU
	model = model.module

	# not train existing layers
	# for k in pretrained_dict:
	count = 0
	para_optim = []
	for k in model.children():
	# for k in model.module.children():
		count += 1
		if count > 6:
			for param in k.parameters():
				para_optim.append(param)
		else:
			for param in k.parameters():
				param.requires_grad = False
		# print(k)
	print('para_optim')
	print(len(para_optim))

	optimizer = optim.RMSprop(para_optim, lr=args.LEARNING_RATE,
							  weight_decay=args.WEIGHT_DECAY, eps=1e-12)
	criterion = nn.NLLLoss2d().cuda()
	exp_dir = args.EXPERIMENT + 'GRU_test'
	if os.path.exists(exp_dir):
		shutil.rmtree(exp_dir)

	exp = experiment.Experiment('GRU_test', args.EXPERIMENT)
	exp.init()

	START_EPOCH = exp.epoch
	END_EPOCH = START_EPOCH + args.N_EPOCHS

	for epoch in range(START_EPOCH, END_EPOCH):

		since = time.time()

		### Train ###
		trn_loss, trn_err = utils.train(model, train_loader, optimizer, criterion, epoch)
		print('Epoch {:d}: Train - Loss: {:.4f}\tErr: {:.4f}'.format(epoch, trn_loss, trn_err))
		time_elapsed = time.time() - since
		print('Train Time {:.0f}m {:.0f}s'.format(
			time_elapsed // 60, time_elapsed % 60))

		### Test ###
		val_loss, val_err = utils.test(model, val_loader, criterion, epoch)
		print('Val - Loss: {:.4f}, Error: {:.4f}'.format(val_loss, val_err))
		time_elapsed = time.time() - since
		print('Total Time {:.0f}m {:.0f}s\n'.format(
			time_elapsed // 60, time_elapsed % 60))

		### Save Metrics ###
		exp.save_history('train', trn_loss, trn_err)
		exp.save_history('val', val_loss, val_err)

		### Checkpoint ###
		exp.save_weights(model, trn_loss, val_loss, trn_err, val_err)
		exp.save_optimizer(optimizer, val_loss)

		## Early Stopping ##
		if (epoch - exp.best_val_loss_epoch) > args.MAX_PATIENCE:
			print(("Early stopping at epoch %d since no "
				   +"better loss found since epoch %.3").format(epoch, exp.best_val_loss))
			break

		# Adjust Lr ###--old method
		utils.adjust_learning_rate(args.LEARNING_RATE, args.LR_DECAY, optimizer,
							 epoch, args.DECAY_LR_EVERY_N_EPOCHS)

		exp.epoch += 1


if __name__=='__main__':
    main()

