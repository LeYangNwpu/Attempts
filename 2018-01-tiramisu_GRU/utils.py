# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torchvision


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
	return round(err, 5)


def train(model, trn_loader, optimizer, criterion, epoch):
	model.train()
	trn_loss = 0
	trn_error = 0
	for batch_idx, (img, targets) in enumerate(trn_loader):
		# inputs = torch.cat((img, cont), dim=1)
		inputs = img
		inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
		optimizer.zero_grad()
		output = model(inputs)
		loss = criterion(output, targets)
		loss.backward()
		optimizer.step()
		trn_loss += loss.data[0]
		pred = get_predictions(output)
		trn_error += error(pred, targets.data.cpu())
	trn_loss /= len(trn_loader)  # n_batches
	trn_error /= len(trn_loader)
	return trn_loss, trn_error


def test(model, test_loader, criterion, epoch=1):
	model.eval()
	test_loss = 0
	test_error = 0
	for img, target in test_loader:
		# data = torch.cat((img, cont), dim=1)
		data = img
		data = Variable(data.cuda(), volatile=True)
		target = Variable(target.cuda())
		output = model(data)
		test_loss += criterion(output, target).data[0]
		pred = get_predictions(output)
		test_error += error(pred, target.data.cpu())
	test_loss /= len(test_loader)  # n_batches
	test_error /= len(test_loader)
	return test_loss, test_error


def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
	"""Sets the learning rate to the initially 
		configured `lr` decayed by `decay` every `n_epochs`"""
	new_lr = lr * (decay ** (cur_epoch // n_epochs))
	for param_group in optimizer.param_groups:
		param_group['lr'] = new_lr


def weights_init(m):
	if isinstance(m, nn.Conv2d):
		# kaiming is first name of author whose last name is 'He' lol
		init.kaiming_uniform(m.weight)
		m.bias.data.zero_()

