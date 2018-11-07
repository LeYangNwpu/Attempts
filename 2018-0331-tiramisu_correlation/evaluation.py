# coding: utf-8

import argparse
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

import dataset_7ch as saliency
import tiramisu_corre as tiramisu
import experiment
import utils_corre as utils


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--EXP_NAME', type=str, default='EncDec_7ch_corre')
	parser.add_argument('--EXP_DIR', type=str, default='/home/yangle/TCyb/result/TrainNet/')
	parser.add_argument('--DATASET_PATH', type=str, default='/home/yangle/TCyb/dataset/tiramisu_7ch_corre/')
	parser.add_argument('--SAVE_DIR', type=str, default='/home/yangle/TCyb/result/mask/EncDec_7ch_corre/')
	parser.add_argument('--LEARNING_RATE', type=float, default=1e-4)
	parser.add_argument('--WEIGHT_DECAY', type=float, default=0.0001)
	args = parser.parse_args()

	if not os.path.exists(args.SAVE_DIR):
		os.makedirs(args.SAVE_DIR)

	normalize = transforms.Normalize(mean=saliency.mean, std=saliency.std)
	test_dset = saliency.TestImage(
		args.DATASET_PATH, 'val', joint_transform=None,
		transform=transforms.Compose([transforms.ToTensor(), normalize, ]))
	test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False)

	model = tiramisu.FcDnSubtle(in_channels=6, n_classes=2)
	model = model.cuda()
	# model = torch.nn.DataParallel(model).cuda()
	optimizer = optim.RMSprop(model.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY)
	# optimizer = optim.Adam(model.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY)
	exper = experiment.Experiment(args.EXP_NAME, args.EXP_DIR)
	exper.resume(model, optimizer)

	count = 1
	for img, name, img_cont, img_box in test_loader:
		img_cont, img_box = Variable(img_cont.cuda()), Variable(img_box.cuda())
		img = Variable(img.cuda())
		output = model(img, img_cont)
		pred = utils.get_predictions(output)
		pred = pred[0]
		img_name = name[0]
		# img_name = str(name)
		# img_name = img_name.replace('tif', 'png')
		save_path = args.SAVE_DIR + img_name
		torchvision.utils.save_image(pred, save_path)
		print(count)
		count += 1


if __name__=='__main__':
    main()



