# coding: utf-8

import argparse
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

import joint_transforms
import saliency_dataset as saliency
import tiramisu as tiramisu
import experiment
import utils


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--EXP_NAME', type=str, default='segment')
	parser.add_argument('--EXP_DIR', type=str, default='/home/yangle/result/TrainNet/')
	parser.add_argument('--DATASET_PATH', type=str, default='/home/yangle/BasicDataset/dataset/MSRA10K/')
	parser.add_argument('--SAVE_DIR', type=str, default='/home/yangle/result/mask/MSRA10K/')
	parser.add_argument('--LEARNING_RATE', type=float, default=1e-4)
	parser.add_argument('--WEIGHT_DECAY', type=float, default=0.0001)
	args = parser.parse_args()

	if not os.path.exists(args.SAVE_DIR):
		os.makedirs(args.SAVE_DIR)

	normalize = transforms.Normalize(mean=saliency.mean, std=saliency.std)
	test_joint_transformer = transforms.Compose([joint_transforms.JointResize(224)])
	test_dset = saliency.TestImage(
		args.DATASET_PATH, 'val', joint_transform=None,
		transform=transforms.Compose([transforms.ToTensor(), normalize, ]))
	test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False)

	model = tiramisu.FCDenseNet57(in_channels=3, n_classes=2)
	# model = model.cuda()
	model = torch.nn.DataParallel(model).cuda()
	optimizer = optim.RMSprop(model.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY)

	exper = experiment.Experiment(args.EXP_NAME, args.EXP_DIR)
	# exper.resume(model, optimizer)
	base_path = args.EXP_DIR + args.EXP_NAME + '/weights/'
	weights_fpath = base_path + 'segment-weights-132-0.109-4.278-0.120-4.493.pth'
	optim_path = base_path + 'segment-optim-132.pth'
	exper.resume(model, optimizer, weights_fpath, optim_path)

	# count = 1
	for count, (img, name) in enumerate(test_loader):
	# for img, name in test_loader:
		data = Variable(img.cuda(), volatile=True)
		output = model(data)
		pred = utils.get_predictions(output)
		pred = pred[0]
		img_name = name[0]
		# img_name = str(name)
		# img_name = img_name.replace('tif', 'png')
		save_path = args.SAVE_DIR + img_name
		torchvision.utils.save_image(pred, save_path)
		print(count)
		# count += 1


if __name__=='__main__':
    main()



