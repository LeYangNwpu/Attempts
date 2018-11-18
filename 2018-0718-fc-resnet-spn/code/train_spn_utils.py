# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from distutils.version import LooseVersion

import jaccard
import f_boundary


def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 233
    #Takes input_loader and returns array of prediction tensors
    predictions = []
    model.eval()
    for input, target in input_loader:
        data, label = Variable(input.cuda(), volatile=True), Variable(target.cuda())
        output = model(data)
        pred = get_predictions(output)
        predictions.append([input,target,pred])
    return predictions


def get_predictions(output_batch):
	# Variables(Tensors) of size (bs,12,224,224)
	bs, c, h, w = output_batch.size()
	tensor = output_batch.data
	# Argmax along channel axis (softmax probabilities)
	values, indices = tensor.cpu().max(1)
	indices = indices.view(bs, h, w)
	return indices


def error(preds, targets):
	assert preds.size() == targets.size()
	bs, h, w = preds.size()
	n_pixels = bs * h * w
	incorrect = preds.ne(targets).cpu().sum()
	err = 100. * incorrect / n_pixels
	# err = err.item()
	return round(err, 5)

############################
# loss function is correct #
############################
# def cross_entropy2d(input, target, weight=None, size_average=True):
#     # input: (n, c, h, w), target: (n, h, w)
#     n, c, h, w = input.size()
#     # log_p: (n, c, h, w)
#     if LooseVersion(torch.__version__) < LooseVersion('0.3'):
#         # ==0.2.X
#         log_p = F.log_softmax(input)
#     else:
#         # >=0.3
#         log_p = F.log_softmax(input, dim=1)
#     # log_p: (n*h*w, c)
#     log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
#     log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
#     log_p = log_p.view(-1, c)
#     # target: (n*h*w,)
#     mask = target >= 0
#     target = target[mask]
#     loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
#     if size_average:
#         loss /= mask.data.sum()
#     return loss


def train(model, trn_loader, optimizer, criterion, epoch):
	model.train()
	trn_loss = 0
	trn_error = 0
	for batch_idx, (inputs, targets) in enumerate(trn_loader):
		inputs = Variable(inputs.cuda())
		optimizer.zero_grad()
		pred_mask = model(inputs)
		target = Variable(targets).cuda()
		loss = criterion(pred_mask, target)
		loss.backward()
		optimizer.step()
		loss_value = loss.data[0]
		print(loss_value)
		trn_loss += loss_value
	trn_loss /= len(trn_loader)  # n_batches
	trn_error /= len(trn_loader)
	return trn_loss, trn_error


def test(model, test_loader, criterion, epoch=1):
	model.eval()
	test_loss = 0
	test_error = 0
	for inputs, targets in test_loader:
		# inputs = Variable(inputs.cuda(), volatile=True)
		# outputs = model(inputs)

		inputs = Variable(inputs.cuda())
		with torch.no_grad():
			outputs = model(inputs)

		pred_mask = outputs[4]
		target = Variable(targets[4]).cuda()
		test_loss += criterion(pred_mask, target)
		pred = get_predictions(pred_mask)
		test_error += error(pred, target.data.cpu())

	test_loss /= len(test_loader)  # n_batches
	test_error /= len(test_loader)
	return test_loss.data[0], test_error


def test_score(model, test_loader):
	model.eval()

	# mask_num = len(test_loader)
	measure_j = np.zeros(1000)
	measure_f = np.zeros(1000)
	count = 0
	for iimg, (inputs, targets) in enumerate(test_loader):
		batch, _, _, _ = inputs.size()
		inputs = Variable(inputs.cuda(), volatile=True)
		pred_mask = model(inputs)
		pred_mask = pred_mask.data.cpu()
		_, indices = pred_mask.max(1)
		for ibat in range(batch):
			target = targets[ibat, :, :, :]
			target = torch.squeeze(target, dim=0)
			target = target.numpy()
			predict = indices[ibat, :, :]
			predict = predict.numpy()

			measure_j[count] = jaccard.db_eval_iou(target, predict)
			measure_f[count] = f_boundary.db_eval_boundary(predict, target)
			count += 1
	mean_j = np.mean(measure_j)
	# measure_j[mask_num] = mean_j
	mean_f = np.mean(measure_f)
	# measure_f[mask_num] = mean_f
	score = (mean_j + mean_f) / 2
	error = 0
	return 1 - score, error



def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
	"""Sets the learning rate to the initially 
		configured `lr` decayed by `decay` every `n_epochs`"""
	new_lr = lr * (decay ** (cur_epoch // n_epochs))
	for param_group in optimizer.param_groups:
		param_group['lr'] = new_lr


def weights_init(m):
	if isinstance(m, nn.Conv2d):
		# kaiming is first name of author whose last name is 'He' lol
		# init.kaiming_uniform_(m.weight)
		init.kaiming_uniform(m.weight)
		# m.bias.data.zero_()

