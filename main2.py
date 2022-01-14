# encoder in first block, linear classifer at end
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as functional, torch.utils.data as torchdata
from torch.autograd import Variable
from dataloaders import dataloaders
from utils import *
from torchsummary import summary
import time, math, numpy as np, matplotlib.pyplot as plt, argparse, os, collections, sys, inspect, pprint, scipy.stats as st
from functools import partial

def get_encoder(dataset, name, nfilters, aeds, aebn, datashape, fixencoder, simpencoder, initialization, testae = False, testloader = None, mean = None, std = None):
	encoder, decoder = create_autoencoder(datashape[1], nfilters, aeds, aebn, False, simpencoder)
	if not fixencoder:
		encoder.apply(initialization)
		return encoder
	encoder_pre, decoder_pre = dataset + '-encoder2-', dataset + '-decoder2-'
	encoder_file = os.path.join(os.getcwd(), 'autoencoders2', encoder_pre + 'weights-all', encoder_pre + name + '.pth')
	decoder_file = os.path.join(os.getcwd(), 'autoencoders2', decoder_pre + 'weights-all', decoder_pre + name + '.pth')
	encoder.load_state_dict(torch.load(encoder_file))
	decoder.load_state_dict(torch.load(decoder_file))
	encoder.eval()
	decoder.eval()
	if testae:
		encoder.to(device)
		decoder.to(device)
		test_autoencoder(datashape, encoder, decoder, testloader, mean, std)
	for param in encoder.parameters():
		param.requires_grad = False
	return encoder

def get_classifier(dataset, name, onecl, clname, apc, fixcl, nclasses, featureshape, bias, initialization):
	if not onecl:
		return None
	classifier = create_classifier(clname, nclasses, featureshape, apc, bias)
	if fixcl:
		cl_file = os.path.join(os.getcwd(), 'classifiers', dataset + clname + '-classifier-' + name + '.pth')
		classifier.load_state_dict(torch.load(cl_file))
		for param in classifier.parameters():
			param.requires_grad = False
		classifier.eval()
	else:
		classifier.apply(initialization)
		classifier.train()
	return classifier

class FirstResBlock_(nn.Module):
	def __init__(self, nfilters, batchnorm, bias, h, first, encoder):
		super(FirstResBlock_, self).__init__()
		self.h = h
		self.first = first
		self.encoder = encoder
		self.batchnorm = batchnorm
		self.cv1 = nn.Conv2d(nfilters, nfilters, 3, 1, 1, bias = bias)
		if self.batchnorm:
			self.bn2 = nn.BatchNorm2d(nfilters)
		self.cv2 = nn.Conv2d(nfilters, nfilters, 3, 1, 1, bias = bias)
	def forward(self, x):
		if self.first:
			x = self.encoder(x)
		z = self.cv1(x)
		z = functional.relu(self.bn2(z)) if self.batchnorm else functional.relu(x)
		z = self.cv2(z)
		return x + self.h * z, z 

class FirstResBlock(nn.Module):
	def __init__(self, nfilters, batchnorm, bias, h):
		super(FirstResBlock, self).__init__()
		self.h = h
		self.batchnorm = batchnorm
		self.cv1 = nn.Conv2d(nfilters, nfilters, 3, 1, 1, bias = bias)
		if self.batchnorm:
			self.bn2 = nn.BatchNorm2d(nfilters)
		self.cv2 = nn.Conv2d(nfilters, nfilters, 3, 1, 1, bias = bias)
	def forward(self, x):
		z = self.cv1(x)
		z = functional.relu(self.bn2(z)) if self.batchnorm else functional.relu(x)
		z = self.cv2(z)
		return x + self.h * z, z 

class ResBlock(nn.Module):
	def __init__(self, nfilters, batchnorm, bias, h):
		super(ResBlock, self).__init__()
		self.h = h
		self.batchnorm = batchnorm
		if self.batchnorm :
			self.bn1 = nn.BatchNorm2d(nfilters)
		self.cv1 = nn.Conv2d(nfilters, nfilters, 3, 1, 1, bias = bias)
		if self.batchnorm :
			self.bn2 = nn.BatchNorm2d(nfilters)
		self.cv2 = nn.Conv2d(nfilters, nfilters, 3, 1, 1, bias = bias)
	def forward(self, x):
		z = functional.relu(self.bn1(x)) if self.batchnorm else functional.relu(x)
		z = self.cv1(z) 
		z = functional.relu(self.bn2(z)) if self.batchnorm else functional.relu(x)
		z = self.cv2(z)
		return x + self.h * z, z

class OneRepResNet(nn.Module):
	def __init__(self, featureshape, first, last, enc, downsample, nclasses, nblocks, bn, bias, smallstep, classifier, clname, apc, initialization):
		super(OneRepResNet, self).__init__()
		h = 1 / nblocks if smallstep else 1
		nfilters = featureshape[1]
		# clname = '1LIN' if last else clname
		self.downsample = downsample
		if self.downsample:
			self.cv = nn.Conv2d(int(nfilters / 2), nfilters, 1, 2, 0, bias = False)
			self.cv.apply(initialization)
		self.stage = nn.ModuleList([FirstResBlock_(nfilters, bn, bias, h, first[1], enc) if first[0] and i == 0 else ResBlock(nfilters, bn, bias, h) for i in range(nblocks)])
		self.stage.apply(initialization)
		if classifier is not None:
			self.classifier = classifier
		else:
			self.classifier = create_classifier(clname, nclasses, featureshape, apc, bias)
			self.classifier.apply(initialization)
	def forward_conv(self, x):
		if self.downsample:
			x = self.cv(x)
		rs = []
		for block in self.stage:
			x, r = block(x)
			rs.append(r)
		return x, rs
	def forward(self, x):
		x, rs = self.forward_conv(x)
		out = self.classifier(x)
		return out, x, rs

class ResNetStage(nn.Module):
	def __init__(self, nblocks, nfilters, first, batchnorm, bias, smallstep):
		super(ResNetStage, self).__init__()
		h = 1 / nblocks if smallstep else 1
		self.blocks = nn.ModuleList([FirstResBlock(nfilters, batchnorm, bias, h) if i == 0 else ResBlock(nfilters, batchnorm, bias, h) for i in range(nblocks)])
	def forward(self, x):
		for block in self.blocks :
			x, _ = block(x)
		return x

class ResNet(nn.Module):
	def __init__(self, downsample, encoder, datashape, nfilters, nclasses, nblocks, batchnorm, bias, smallstep, classifier, clname, apc, initialization):
		super(ResNet, self).__init__()
		self.downsample =downsample
		self.encoder = encoder
		self.nstages = int(nblocks / downsample) if downsample else 1
		self.nblocks = downsample if downsample else nblocks
		self.stages = nn.ModuleList([ResNetStage(self.nblocks, 2 ** i * nfilters, 1, batchnorm, bias, smallstep) for i in range(self.nstages)])
		self.stages.apply(initialization)
		if self.downsample:
			self.downsamples = nn.ModuleList([nn.Conv2d(2 ** i * nfilters, 2 ** (i + 1) * nfilters, 1, 2, 0, bias = False) for i in range(self.nstages - 1)])
			self.downsamples.apply(initialization)
		with torch.no_grad(): 
			featureshape = list(self.forward_conv(torch.ones(*datashape)).shape)
		if classifier is not None:
			self.classifier = classifier
		else:
			self.classifier = create_classifier(clname, nclasses, featureshape, apc, bias)
			self.classifier.apply(initialization)
	def forward_conv(self, x):
		x = self.encoder(x)
		for i in range(self.nstages):
			x = self.stages[i](x)
			if self.downsample and i < self.nstages - 1:
				x = self.downsamples[i](x) 
		return x
	def forward(self, x):
		x = self.forward_conv(x)
		x = self.classifier(x)
		return x

class ResNextBlock(nn.Module):
	def __init__(self, avg, featureshape, infilters = 256, planes = 64, expansion = 4, cardinality = 32, width = 4, base = 64, stride = 1, relu = True, residu = True, 
				 downsample = None, clname = '1LIN', bias = 0, nclasses = 100):
		super(ResNextBlock, self).__init__()
		self.relu = relu 
		self.residu = residu
		self.clname = clname
		self.intfilters = cardinality * math.floor(planes * width / base)
		self.outfilters = planes * expansion
		self.cv1 = nn.Conv2d(infilters, self.intfilters, 1, 1, 0, bias = False)
		self.bn1 = nn.BatchNorm2d(self.intfilters)
		self.cv2 = nn.Conv2d(self.intfilters, self.intfilters, 3, stride, 1, groups = cardinality, bias = False)
		self.bn2 = nn.BatchNorm2d(self.intfilters)
		self.cv3 = nn.Conv2d(self.intfilters, self.outfilters, 1, 1, 0, bias = False)
		self.bn3 = nn.BatchNorm2d(self.outfilters)
		self.downsample = downsample
		self.avgpool = nn.AvgPool2d(avg, 1)
		self.classifier = create_classifier(clname, nclasses, featureshape, avg, bias)
	def forward(self, x):
		r = functional.relu(self.bn1(self.cv1(x)), inplace = True)
		r = functional.relu(self.bn2(self.cv2(r)), inplace = True)
		r = functional.relu(self.bn3(self.cv3(r)), inplace = True)
		x = self.downsample(x) if self.downsample is not None else x
		z = functional.relu(x + r, inplace = True) if self.relu else x + r
		r = z - x if self.relu and self.residu else r
		w = self.avgpool(z) if self.clname[1:] == 'LIN' else z
		out = self.classifier(w)
		return out, z, [r]

class ResNextBlock_(nn.Module):
	def __init__(self, infilters = 256, planes = 64, expansion = 4, cardinality = 32, width = 4, base = 64, stride = 1, relu = True, residu = True, downsample = None):
		super(ResNextBlock, self).__init__()
		self.relu = relu 
		self.residu = residu
		self.intfilters = cardinality * math.floor(planes * width / base)
		self.outfilters = planes * expansion
		self.cv1 = nn.Conv2d(infilters, self.intfilters, 1, 1, 0, bias = False)
		self.bn1 = nn.BatchNorm2d(self.intfilters)
		self.cv2 = nn.Conv2d(self.intfilters, self.intfilters, 3, stride, 1, groups = cardinality, bias = False)
		self.bn2 = nn.BatchNorm2d(self.intfilters)
		self.cv3 = nn.Conv2d(self.intfilters, self.outfilters, 1, 1, 0, bias = False)
		self.bn3 = nn.BatchNorm2d(self.outfilters)
		self.downsample = downsample
	def forward(self, x):
		r = functional.relu(self.bn1(self.cv1(x)), inplace = True)
		r = functional.relu(self.bn2(self.cv2(r)), inplace = True)
		r = functional.relu(self.bn3(self.cv3(r)), inplace = True)
		if self.downsample is not None:
			x = self.downsample(x)
		if self.relu :
			z = functional.relu(x + r, inplace = True)
			if self.residu :
				r = z - x
		else :
			z = x + r 
		return z, r

class ResNextStage(nn.Module):
	def __init__(self, nb, inf = 256, pln = 64, exp = 4, card = 32, width = 4, base = 64, stride = 1, rel = True, res = True):
		super(ResNextStage, self).__init__()
		intf = pln * exp
		ds = nn.Sequential(nn.Conv2d(inf, intf, 1, stride, bias = False), nn.BatchNorm2d(intf)) if stride != 1 or inf != intf else None
		block = lambda i : ResNextBlock_(inf, pln, exp, card, width, base, stride, rel, res, ds) if i == 0 else ResNextBlock_(intf, pln, exp, card, width, base, 1, rel, res)
		self.blocks = nn.ModuleList([block(i) for i in range(nb)])
	def forward(self, x):
		rs = []
		for block in self.blocks :
			x, r = block(x)
			rs.append(r)
		return x, rs

class ResNext29(nn.Module):
	def __init__(self, datashape, nclasses, learnencoder, encoder, initialization, nblocks = [3, 3, 3], infilters = 64, planes = 64, expansion = 4, 
				 cardinality = 16, width = 64, base = 64, relu = True, residu = True):
		super(ResNext29, self).__init__()
		self.encoder = encoder
		self.stage1 = ResNextStage(nblocks[0], infilters * 1, planes * 1, expansion, cardinality, width, base, 1, relu, residu)
		self.stage2 = ResNextStage(nblocks[1], infilters * 4, planes * 2, expansion, cardinality, width, base, 2, relu, residu)
		self.stage3 = ResNextStage(nblocks[2], infilters * 8, planes * 4, expansion, cardinality, width, base, 2, relu, residu)
		self.avgpool = nn.AvgPool2d(7, 1)
		with torch.no_grad():
			featuresize = self.forward_conv(torch.zeros(*datashape))[0].view(-1).shape[0]
		self.classifier = nn.Linear(featuresize, nclasses)
		if not learnencoder:
			for param in self.encoder.parameters():
				param.requires_grad = False
	def forward_conv(self, x):
		rs = dict()
		x = self.encoder(x)
		x, rs[1] = self.stage1(x)
		x, rs[2] = self.stage2(x)
		x, rs[3] = self.stage3(x)
		x = self.avgpool(x)
		return x, [r for i in range(1, 4) for r in rs[i]]
	def forward(self, x):
		x, rs = self.forward_conv(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x, rs

class ResNext50(nn.Module):
	def __init__(self, datashape, nclasses, learnencoder, encoder, initialization, nblocks = [3, 4, 6, 3], infilters = 64, planes = 64, expansion = 4, 
				 cardinality = 32, width = 4, base = 64, relu = True, residu = True):
		super(ResNext50, self).__init__()
		self.encoder = encoder
		self.stage1 = ResNextStage(nblocks[0], infilters * 1, planes * 1, expansion, cardinality, width, base, 1, relu, residu)
		self.stage2 = ResNextStage(nblocks[1], infilters * 4, planes * 2, expansion, cardinality, width, base, 2, relu, residu)
		self.stage3 = ResNextStage(nblocks[2], infilters * 8, planes * 4, expansion, cardinality, width, base, 2, relu, residu)
		self.stage4 = ResNextStage(nblocks[3], infilters * 16, planes * 8, expansion, cardinality, width, base, 2, relu, residu)
		self.avgpool = nn.AvgPool2d(7 if datashape[-1] == 224 else 4, 1)
		with torch.no_grad():
			featuresize = self.forward_conv(torch.zeros(*datashape))[0].view(-1).shape[0]
		self.classifier = nn.Linear(featuresize, nclasses)
		if not learnencoder:
			for param in self.encoder.parameters():
				param.requires_grad = False
	def forward_conv(self, x):
		rs = dict()
		x = self.encoder(x)
		x, rs[1] = self.stage1(x)
		x, rs[2] = self.stage2(x)
		x, rs[3] = self.stage3(x)
		x, rs[4] = self.stage4(x)
		x = self.avgpool(x)
		return x, [r for i in range(1, 5) for r in rs[i]]
	def forward(self, x):
		x, rs = self.forward_conv(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x, rs

class Encoder(nn.Module):
	def __init__(self, encoder, featureshape, nclasses, nfilters, classifier, clname, apc, bias, initialization):
		super(Encoder, self).__init__()
		self.encoder = encoder
		if classifier is not None:
			self.classifier = classifier
		else:
			self.classifier = create_classifier(clname, nclasses, featureshape, apc, bias)
			self.classifier.apply(initialization)
	def forward(self, x):
		x = self.encoder(x)
		out = self.classifier(x)
		return out, x, []

def get_resnet_blocks(downsample, enc, encodingshape, nclasses, nmodels, nbl, batchnorm, bias, smallstep, classifier, clname, apc, init):
	f = lambda i : 2 ** int(i / downsample) 
	ds = lambda i : i and downsample and not i % downsample 
	featshape = lambda i : encodingshape if not downsample else [1, encodingshape[1] * f(i), int(encodingshape[2] / f(i)), int(encodingshape[3] / f(i))]
	first = lambda i : (not i or ds(i), not i)
	last = lambda i : i == nmodels - 1
	return [OneRepResNet(featshape(i), first(i), last(i), enc, ds(i), nclasses, nbl, batchnorm, bias, smallstep, classifier, clname, apc, init) for i in range(nmodels)]

def resnext_stage(nb, avg, featureshape, inf = 256, pln = 64, exp = 4, card = 32, width = 4, base = 64, stride = 1, rel = True, res = True):
	intf = pln * exp
	ds = lambda i : nn.Sequential(nn.Conv2d(inf, intf, 1, stride, bias = False), nn.BatchNorm2d(intf)) if i == 0 and (stride != 1 or inf != intf) else None
	strd = lambda i : stride if i == 0 else 1 
	infilters = lambda i : inf if i == 0 else intf
	return [ResNextBlock(avg, featureshape, infilters(i), pln, exp, card, width, base, strd(i), rel, res, ds(i)) for i in range(nb)]

def get_resnext_blocks(infilters = 64, planes = 64, expansion = 4, cardinality = 32, width = 4, base = 64, rel = True, res = True, initialization = None):
	nbs, avgs, nfls, dims, models = [3, 4, 6, 3], [15, 11, 7, 3], [256, 512, 1024, 2048], [32, 16, 8, 4], []
	fct_infilters, fct_planes, strides = [1, 4, 8, 16], [1, 2, 4, 8], [1, 2, 2, 2]
	feature_shape = lambda f, h, w, k, s : (f, (h - k) / s + 1, (w - k) / s + 1)
	for i in range(4):
		featshape = feature_shape(nfls[i], dims[i], dims[i], avgs[i], 1)
		featsize = int(np.prod(featshape))
		models += resnext_stage(nbs[i], avgs[i], featshape, infilters * fct_infilters[i], planes * fct_planes[i], expansion, cardinality, width, base, strides[i], rel, res)
	for model in models:
		model.apply(initialization)
	return models

def get_models(modelname, initialization, traintype, enc, downsample = 0, encodingshape = None, nclasses = 10, nmodels = 10, nblocks = 1, batchnorm = 1, bias = 0, 
			   smallstep = 0, classifier = None, clname = '3Lin', avgpoolcl = 0, rel = True, res = True):
	if modelname == 'resnext50':
		return get_resnext_blocks(rel = rel, res = res, initialization = initialization)
	if modelname == 'resnet':
		return get_resnet_blocks(downsample, enc, encodingshape, nclasses, nmodels, nblocks, batchnorm, bias, smallstep, classifier, clname, avgpoolcl, initialization)

def train_submodel(totrain, models, nblocks, optimizers, schedulers, criterion, tra, mean, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne1, ne2, 
				   trainloader, valloader, testloader, r = None):
	print('\n' + '-' * 64, f'Round {r}' if r is not None else '', 'Submodel', totrain)
	train_loss, train_accuracy, val_accuracy, it = [], [], [], 0
	if lml0type == 'decreasing':
		lml, lmt = lml0 / totrain ** lml0power if totrain > 0 else lml0, totrain ** lml0power / lml0
	elif lml0type == 'increasing':
		lml, lmt =  lml0 * totrain ** lml0power, 1 / (lml0 * totrain ** lml0power) if totrain > 0 else 1 / lml0
	for epoch in range(1, ne1 + totrain * ne2 + 1):
		for mod in models:
			mod.train()
		t1, loss_meter, accuracy_meter = time.time(), AverageMeter(), AverageMeter()
		for j, (x, y) in enumerate(trainloader):
			it = it + 1
			x, y = x.to(device), y.to(device)
			z = Variable(x.data, requires_grad = False).detach()
			for i in range(totrain + 1):
				optimizers[i].zero_grad()
				out, w, rs = models[i](z)
				z = Variable(w.data, requires_grad = False).detach()
				if i == totrain:
					target = criterion(out, y)
					if (tra or uza) and i > 0 :
						transport = sum([torch.mean(r ** 2) for r in rs]) if mean else sum([torch.sum(r ** 2) for r in rs]) / (z.shape[0] * nblocks)
					if i > 0 and uza and it % uzs == 0 :
						lml += uzt * target.item()
						lmt = 1 / lml
					loss = (target + transport / (2 * taus[i]) if tra else (target + lmt * transport if uza else target)) if i > 0 else target
					loss.backward()
					optimizers[i].step()
					if schedulers is not None:
						schedulers[i].step()
					_, pred = torch.max(out.data, 1)
					update_meters(y, pred, target.item(), loss_meter, accuracy_meter)
		epoch_train_loss, epoch_train_accuracy, epoch_val_accuracy = loss_meter.avg, accuracy_meter.avg, test_submodel(totrain, models, criterion, testloader)
		print('\n' + '-' * 64, f'Round {r}' if r is not None else '', 'Submodel', totrain, 'Epoch', epoch, 'Took', time.time() - t1, 's')
		print('Transport', tra, 'tau =', taus[totrain], 'Uzawa', uza, 'lmt =', lmt)
		print('Train loss', epoch_train_loss, 'Train accuracy', epoch_train_accuracy, 'Val accuracy', epoch_val_accuracy)
		train_loss.append(epoch_train_loss)
		train_accuracy.append(epoch_train_accuracy)
		val_accuracy.append(epoch_val_accuracy)
	return train_loss, val_accuracy

def train_seq(models, nblocks, optimizers, schedulers, criterion, tra, mean, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne1, ne2, 
			  trainloader, valloader, testloader, r = None):
	train_loss, train_accuracy, val_accuracy = [], [], []
	for totrain in range(len(models)):
		trloss, vlacc = train_submodel(totrain, models, nblocks, optimizers, schedulers, criterion, tra, mean, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne1, ne2, 
									   trainloader, valloader, testloader, r)
	return trloss, vlacc

def test_submodel(totest, models, criterion, loader):
	loss_meter, accuracy_meter = AverageMeter(), AverageMeter()
	for mod in models:
		mod.eval()
	for j, (x, y) in enumerate(loader):
		x, y = x.to(device), y.to(device)
		z = Variable(x.data, requires_grad = False).detach()
		for i in range(totest + 1) :
			with torch.no_grad():
				out, w, rs = models[i](z)
				z = Variable(w.data, requires_grad = False).detach()
				if i == totest:
					target = criterion(out, y)
					_, pred = torch.max(out.data, 1)
					update_meters(y, pred, target.item(), loss_meter, accuracy_meter)
	return accuracy_meter.avg

def train_par(models, nblocks, optimizers, schedulers, criterion, tra, mean, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne0, trainloader, valloader, testloader):
	t0, nummodels, train_loss, train_accuracy, val_accuracy, it = time.time(), len(models), [], [], [], 0 
	if lml0type == 'decreasing':
		lmls, lmts = [lml0 / i ** lml0power if i > 0 else lml0 for i in range(nummodels)], [i ** lml0power / lml0 for i in range(nummodels)] 
	elif lml0type == 'increasing':
		lmls, lmts =  [lml0 * i ** lml0power for i in range(nummodels)], [1 / (lml0 * i ** lml0power) if i > 0 else 1 / lml0 for i in range(nummodels)] 
	for epoch in range(1, ne0 + 1):
		for mod in models:
			mod.train()
		t1, loss_meters, accuracy_meters = time.time(), [AverageMeter() for _ in range(nummodels)], [AverageMeter() for _ in range(nummodels)]
		for j, (x, y) in enumerate(trainloader):
			it = it + 1
			x, y = x.to(device), y.to(device)
			z = Variable(x.data, requires_grad = False).detach()
			for i, model in enumerate(models):
				optimizers[i].zero_grad()
				out, w, rs = model(z)
				z = Variable(w.data, requires_grad = False).detach()
				target = criterion(out, y)
				if (tra or uza) and i > 0 :
					transport = sum([torch.mean(r ** 2) for r in rs]) if mean else sum([torch.sum(r ** 2) for r in rs]) / (z.shape[0] * nblocks)
				if i > 0 and uza and it % uzs == 0 :
					lmls[i] += uzt * target.item()
					lmts[i] = 1 / lmls[i]
				loss = (target + transport / (2 * taus[i]) if tra else (target + lmts[i] * transport if uza else target)) if i > 0 else target
				loss.backward()
				optimizers[i].step()
				if schedulers is not None:
					schedulers[i].step()
				_, pred = torch.max(out.data, 1)
				update_meters(y, pred, target.item(), loss_meters[i], accuracy_meters[i])
		epoch_val_accuracies = test_par(models, criterion, testloader)
		epoch_train_losses, epoch_train_accuracies = [loss_meters[i].avg for i in range(nummodels)], [accuracy_meters[i].avg for i in range(nummodels)]
		print('-' * 64, 'Epoch', epoch, 'took', time.time() - t1, 's')
		print('Transport', tra, 'taus', taus, 'Uzawa', uza, 'lmts', lmts)
		print('Train losses', epoch_train_losses, '\nTrain accuracies', epoch_train_accuracies, '\nVal accuracies', epoch_val_accuracies)
		train_loss.append(np.max(epoch_train_losses))
		train_accuracy.append(np.max(epoch_train_accuracies))
		val_accuracy.append(np.max(epoch_val_accuracies))
	return train_loss, val_accuracy

def test_par(models, criterion, loader):
	nummodels = len(models)
	loss_meters, accuracy_meters = [AverageMeter() for _ in range(nummodels)], [AverageMeter() for _ in range(nummodels)]
	for model in models:
		model.eval()
	for j, (x, y) in enumerate(loader):
		x, y = x.to(device), y.to(device)
		z = Variable(x.data, requires_grad = False).detach()
		for i, model in enumerate(models):
			with torch.no_grad():
				out, w, rs = model(z)
				z = Variable(w.data, requires_grad = False).detach()
				target = criterion(out, y)
				_, pred = torch.max(out.data, 1)
				update_meters(y, pred, target.item(), loss_meters[i], accuracy_meters[i])
	return [accuracy_meters[i].avg for i in range(nummodels)]

def train_mro(models, nblocks, optimizers, schedulers, criterion, tra, mean, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne1, ne2, nrounds, 
			  trainloader, valloader, testloader):
	for r in range(1, nrounds + 1):
		print('\n' + '-' * 64, 'Round', r)
		trloss, vlacc = train_seq(models, nblocks, optimizers, schedulers, criterion, tra, mean, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne1, ne2, 
								  trainloader, valloader, testloader, r)
	return trloss, vlacc

def train_e2e(model, opt, optimizer, scheduler, criterion, nepochs, trainloader, valloader, testloader):
	t0, train_loss, val_accuracy = time.time(), [], []
	print('\n--- Begin e2e trainning\n')
	for e in range(nepochs):
		model.train()
		t1, loss_meter, accuracy_meter = time.time(), AverageMeter(), AverageMeter()
		for j, (x, y) in enumerate(trainloader):
			x, y = x.to(device), y.to(device)
			optimizer.zero_grad()
			out = model(x)
			loss = criterion(out, y) 
			loss.backward()
			optimizer.step()
			_, pred = torch.max(out.data, 1)
			update_meters(y, pred, loss.item(), loss_meter, accuracy_meter)
		if scheduler is not None:
			scheduler.step()
		epoch_val_acc = test_e2e(model, criterion, testloader)
		train_loss.append(loss_meter.avg)
		val_accuracy.append(epoch_val_acc)
		m = (e + 1, nepochs, loss_meter.avg, epoch_val_acc, time.time() - t1, time.time() - t0)
		print('\n[***** Ep {:^5}/{:^5} over ******] Train loss {:.4f} Valid acc {:.4f} Epoch time {:9.4f}s Total time {:.4f}s\n'.format(*m))
	return train_loss, val_accuracy

def test_e2e(model, criterion, loader):
	model.eval()
	loss_meter, accuracy_meter = AverageMeter(), AverageMeter()
	for j, (x, y) in enumerate(loader):
		with torch.no_grad():
			x, y = x.to(device), y.to(device)
			out = model(x)
			loss = criterion(out, y)
			_, pred = torch.max(out.data, 1)
			update_meters(y, pred, loss.item(), loss_meter, accuracy_meter)
	return accuracy_meter.avg

def train_blockwise(traintype, models, nblocks, optimizers, schedulers, criterion, tra, mean, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne0, ne1, ne2, nrounds, 
				    trainloader, valloader, testloader):
	if traintype == 'seq':
		return train_seq(models, nblocks, optimizers, schedulers, criterion, tra, mean, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne1, ne2, 
						 trainloader, valloader, testloader)
	if traintype == 'par':
		return train_par(models, nblocks, optimizers, schedulers, criterion, tra, mean, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne0, 
			   		     trainloader, valloader, testloader)
	elif traintype == 'mro':
		return train_mro(models, nblocks, optimizers, schedulers, criterion, tra, mean, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne1, ne2, nrounds, 
						 trainloader, valloader, testloader)

def end2end_exp(modname, downsample, encoder, datashape, nfilters, nclasses, nmodels, nblocks, batchnorm, bias, smallstep, classifier, clname, avgpoolcl, 
			    initialization, opt, lrt, lrd, be1, be2, nepochs0, trainloader, valloader, testloader):
	if modname == 'resnet':
		model = ResNet(downsample, encoder, datashape, nfilters, nclasses, nmodels * nblocks, batchnorm, bias, smallstep, classifier, clname, avgpoolcl, initialization)
	elif modname == 'resnext50':
		model = ResNext50(datashape, nclasses, learnencoder, encoder, initialization)
	m, w = 0.9, 0.0001
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr = lrt, momentum = m, weight_decay = w) if opt == 'sgd' else optim.Adam(model.parameters(), lr = lrt, betas = (be1, be2))
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [120, 160, 200], gamma = 0.2) if opt == 'sgd' and lrd else None
	model.to(device)
	print(model)
	summary(model, datashape[1:])
	train_loss, val_accuracy = train_e2e(model, opt, optimizer, scheduler, criterion, nepochs0, trainloader, valloader, testloader)
	del model
	return train_loss, val_accuracy

def blockwise_exp(modname, nfilters, traintype, enc, encodingshape, classifier, clname, apc, nclasses, nmodels, nblocks, downsample, bn, bias, smallstep, rel, res,
 				  initialization, opt, learningrate, learningratedecay, beta1, beta2, transport, mean, tau, varyingtau, lambdaloss0, lambdaloss0type, lambdaloss0power, 
 				  uzawatau, uzawasteps, uzawa, nepochs0, nepochs1, nepochs2, nrounds, trainloader, valloader, testloader):
	models = get_models(modname, initialization, traintype, enc, downsample, encodingshape, nclasses, nmodels, nblocks, bn, bias, smallstep, classifier, clname, apc, rel, res) 
	if opt == 'adam':
		optimizers = [optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr = learningrate, betas = (beta1, beta2)) for model in models] 
		schedulers = None
	elif opt == 'sgd':
		optimizers = [optim.SGD(filter(lambda p : p.requires_grad, model.parameters()), lr = learningrate, momentum = 0.9, weight_decay = 0.0001) for model in models] 
		schedulers = [optim.lr_scheduler.MultiStepLR(optimizer, milestones = [120, 160, 200], gamma = 0.2) for optimizer in optimizers] if learningratedecay else None
	taus = [0] + [tau / 2] * int(nmodels / 2) + [tau] * int(nmodels / 2) if varyingtau else [0] + [tau] * nmodels
	criterion = nn.CrossEntropyLoss()
	for model in models:
		model.to(device)
	# check_weights(models, clname)
	train_loss, val_accuracy = train_blockwise(traintype, models, nblocks, optimizers, schedulers, criterion, transport, mean, taus, uzawa, lambdaloss0, lambdaloss0type, 
											   lambdaloss0power, uzawatau, uzawasteps, nepochs0, nepochs1, nepochs2, nrounds, trainloader, valloader, testloader)
	# check_weights(models, clname)
	for model in models:
		del model
	return train_loss, val_accuracy
	
def check_weights(models, clname):
	j = 1 if clname[1:] == 'LIN' else 0
	print('encoder weight', models[0].encoder[0].weight[0,0,0])
	print('classifier 00 weight', models[0].classifier[j].weight[0,0])
	print('classifier 01 weight', models[1].classifier[j].weight[0,0])
	print('classifier -1 weight', models[-1].classifier[j].weight[0,0])

def experiment(dataset, modelname, batchsize, nfilters, traintype, aeds, aebn, fixencoder, simpencoder, testae, clname, avgpoolcl, onecl, fixcl, nmodels, nblocks, 
			   downsample, batchnorm, bias, smallstep, relu, residu, initname, initgain, optimizer, learningrate, learningratedecay, beta1, beta2, transport, mean, 
			   tau, varyingtau, lambdaloss0, lambdaloss0type, lambdaloss0power, uzawatau, uzawasteps, nepochs0, nepochs1, nepochs2, nrounds, trainsize, valsize, 
			   testsize, seed, experiments):

	t0 = time.time()
	if seed is not None:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.manual_seed(seed)
		np.random.seed(seed)
	uzawa = 1 if (uzawatau > 0 and uzawasteps > 0) else 0
	nepochs = nepochs0 if traintype in ['par', 'e2e'] else nepochs1
	if downsample:
		onecl, fixcl = 0, 0
		print('onecl and fixcl set to False because of downsampling')
	if simpencoder:
		fixencoder = 0
		print('fixencoder set to False because simpencoder')
	if varyingtau and tau > 0 and not transport:
		transport = 1
		print('transport set to True because varyingtau')
	if not transport and tau > 0:
		tau = 0
		print('no transport despite tau > 0 because transport is False')
	if transport and uzawa:
		transport = 0
		print('transport set to False because uzawa')
	if modelname == 'resnext50':
		aeds, aebn, onecl, fixcl, fixencoder, dataset, nmodels, nfilters = False, True, False, False, False, 'cifar100', 16, 64
		print('resnext so aeds, fixencoder, onecl and fixcl set to False, aebn to True, nmodels to 16, nfilters to 64, dataset to cifar100')
	if experiments and nepochs > 1:
		expname = [f'trs{trainsize}', f'tra{transport}', f'mea{mean}', f'uza{uzawa}', f'vta{varyingtau}', f'tau{tau}']
		if modelname == 'resnet':
			expname = [f'sen{simpencoder}', f'ocl{onecl}', f'apc{avgpoolcl}', f'nfl{nfilters}', f'nmo{nmodels}', f'nbl{nblocks}', f'dow{downsample}'] + expname
		elif modelname == 'resnext50':
			expname = [f'rel{relu}', f'res{residu}'] + expname
		if uzawa:
			expname = expname + [f'lml0{lambdaloss0}', lambdaloss0type, f'lml0p{lambdaloss0power}', f'uzt{uzawatau}', f'uzs{uzawasteps}']
		stdout0 = sys.stdout
		d = {'par': str(nepochs0), 'e2e': str(nepochs0), 'seq': str(nepochs1) + '-' + str(nepochs2), 'mro': str(nrounds) + '-' + str(nepochs1) + '-' + str(nepochs2)}
		expname = ['log', 'gflowall15', modelname, dataset, traintype, d[traintype], clname, initname, f'ing{initgain}', optimizer, f'lrt{learningrate}', 
				   f'lrd{learningratedecay}'] + expname
		sys.stdout = open('-'.join(expname + [time.strftime("%Y%m%d-%H%M%S")]) + '.txt', 'wt')

	frame = inspect.currentframe()
	names, _, _, values = inspect.getargvalues(frame)
	print('experiment from gflowall15.py with parameters')
	for name in names:
		print('%s = %s' % (name, values[name]))

	pretrainedname = ('ds' if aeds else '') + ('bn' if aebn else '') + str(nfilters) + ('trs' + str(trainsize) if trainsize is not None else '')
	trainloader, valloader, testloader, datashape, nclasses, datamean, datastd = dataloaders(dataset, batchsize, trainsize, valsize, testsize)
	initialization = partial(initialize, initname, initgain)
	encoder = get_encoder(dataset, pretrainedname, nfilters, aeds, aebn, datashape, fixencoder, simpencoder, initialization, testae, testloader, datamean, datastd)
	encodingshape = list(encoder(torch.ones(*datashape).to(device)).shape) if testae and fixencoder else list(encoder(torch.ones(*datashape)).shape)
	classifier = get_classifier(dataset, pretrainedname, onecl, clname, avgpoolcl, fixcl, nclasses, encodingshape, bias, initialization) 
	print('train batches', len(trainloader), 'val batches', len(valloader), 'batchsize', batchsize)
	if traintype == 'e2e':
		trloss, vlacc =  end2end_exp(modelname, downsample, encoder, datashape, nfilters, nclasses, nmodels, nblocks, batchnorm, bias, smallstep, classifier, clname, 
									 avgpoolcl, initialization, optimizer, learningrate, learningratedecay, beta1, beta2, nepochs0, trainloader, valloader, testloader)
	else:
		trloss, vlacc =  blockwise_exp(modelname, nfilters, traintype, encoder, encodingshape, classifier, clname, avgpoolcl, nclasses, nmodels, nblocks, downsample, 
									   batchnorm, bias, smallstep, relu, residu, initialization, optimizer, learningrate, learningratedecay, beta1, beta2, transport, 
									   mean, tau, varyingtau, lambdaloss0, lambdaloss0type, lambdaloss0power, uzawatau, uzawasteps, uzawa, nepochs0, nepochs1, nepochs2, 
									   nrounds, trainloader, valloader, testloader)

	if experiments and nepochs > 1:
		print('--- train loss \n', trloss, '\n--- val acc \n', vlacc)
		print('--- min train loss \n', min(trloss), '\n--- max val acc \n', max(vlacc))
		sys.stdout.close()
		sys.stdout = stdout0
	return trloss, vlacc, time.time() - t0

def experiments(parameters, average):
	t0, j, f, accs, nparameters = time.time(), 0, 110, [], len(parameters) 
	nexperiments = int(np.prod([len(parameters[i][1]) for i in range(nparameters)]))
	sep = '-' * f 
	print('\n' + sep, 'gflowall15.py')
	print(sep, nexperiments, 'gflowall15 experiments ' + ('to average ' if average else '') + 'over parameters:')
	pprint.pprint(parameters, width = f, compact = True)
	for params in product([values for name, values in parameters]) :
		j += 1
		print('\n' + sep, 'gflowall15 experiment %d/%d with parameters:' % (j, nexperiments))
		pprint.pprint([parameters[i][0] + ' = ' + str(params[i]) for i in range(nparameters)], width = f, compact = True)
		train_loss, val_accuracy, t1 = experiment(*params, True)
		accs.append(np.max(val_accuracy))
		print(sep, 'gflowall15 experiment %d/%d over. took %.1f s. total %.1f s' % (j, nexperiments, t1, time.time() - t0))
	if average:
		acc = np.mean(accs)
		confint = st.t.interval(0.95, len(accs) - 1, loc = acc, scale = st.sem(accs))
		print('\nall val acc', accs)
		print('\naverage val acc', acc)
		print('\nconfint', confint)
	print(('\n' if not average else '') + sep, 'total time for %d experiments: %.1f s' % (j, time.time() - t0))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-dat", "--dataset", required = True, choices = ['mnist', 'cifar10', 'cifar100'], nargs = '*')
	parser.add_argument("-mod", "--modelname", required = True, choices = ['resnext50', 'resnet'], nargs = '*')
	parser.add_argument("-bas", "--batchsize", type = int, default = [128], nargs = '*')
	parser.add_argument("-nfl", "--nfilters", type = int, default = [32], nargs = '*')
	parser.add_argument("-trt", "--traintype", default = ['seq'], choices = ['seq', 'par', 'mro', 'e2e'], nargs = '*')
	parser.add_argument("-ads", "--aeds", type = int, default = [1], choices = [0, 1], nargs = '*')
	parser.add_argument("-abn", "--aebn", type = int, default = [1], choices = [0, 1], nargs = '*')
	parser.add_argument("-fen", "--fixencoder", type = int, default = [0], choices = [0, 1], nargs = '*')
	parser.add_argument("-sen", "--simpleencoder", type = int, default = [0], choices = [0, 1], nargs = '*')
	parser.add_argument("-tae", "--testae", type = int, default = [0], choices = [0, 1], nargs = '*')
	parser.add_argument("-cln", "--clname", default = ['3LIN'], choices = ['1LIN', '2LIN', '3LIN', '1CNN'], nargs = '*')
	parser.add_argument("-apc", "--avgpoolcl", type = int, default = [0], nargs = '*')
	parser.add_argument("-ocl", "--onecl", type = int, default = [0], choices = [0, 1], nargs = '*')
	parser.add_argument("-fcl", "--fixcl", type = int, default = [0], choices = [0, 1], nargs = '*')
	parser.add_argument("-nmo", "--nmodels", type = int, default = [10], nargs = '*')
	parser.add_argument("-nbl", "--nblocks", type = int, default = [1], nargs = '*')
	parser.add_argument("-dow", "--downsample", type = int, default = [0], nargs = '*')
	parser.add_argument("-btn", "--batchnorm", type = int, default = [1], choices = [0, 1], nargs = '*')
	parser.add_argument("-bia", "--bias", type = int, default = [0], choices = [0, 1], nargs = '*')
	parser.add_argument("-sms", "--smallstep", type = int, default = [0], choices = [0, 1], nargs = '*')
	parser.add_argument("-rel", "--relu", type = int, default = [1], choices = [0, 1], nargs = '*')
	parser.add_argument("-res", "--residu", type = int, default = [1], choices = [0, 1], nargs = '*')
	parser.add_argument("-inn", "--initname", default = ['orthogonal'], choices = ['orthogonal', 'normal', 'kaiming'], nargs = '*')
	parser.add_argument("-ing", "--initgain", type = float, default = [0.05], nargs = '*')
	parser.add_argument("-opt", "--optimizer", default = ['sgd'], choices = ['adam', 'sgd'], nargs = '*')
	parser.add_argument("-lrt", "--learningrate", type = float, default = [0.007], nargs = '*')
	parser.add_argument("-lrd", "--learningratedecay", type = int, default = [0], choices = [0, 1], nargs = '*')
	parser.add_argument("-be1", "--beta1", type = float, default = [0.9], nargs = '*')
	parser.add_argument("-be2", "--beta2", type = float, default = [0.999], nargs = '*')
	parser.add_argument("-tra", "--transport", type = int, default = [0], choices = [0, 1], nargs = '*')
	parser.add_argument("-mea", "--mean", type = int, default = [1], choices = [0, 1], nargs = '*')
	parser.add_argument("-tau", "--tau", type = float, default = [0], nargs = '*')
	parser.add_argument("-vta", "--varyingtau", type = int, default = [0], choices = [0, 1], nargs = '*')
	parser.add_argument("-lml", "--lambdaloss0", type = float, default = [1], nargs = '*')
	parser.add_argument("-lmt", "--lambdaloss0type", default = ['increasing'], choices = ['increasing', 'decreasing'], nargs = '*')
	parser.add_argument("-lmp", "--lambdaloss0power", type = float, default = [1], nargs = '*')
	parser.add_argument("-uzt", "--uzawatau", type = float, default = [0], nargs = '*')
	parser.add_argument("-uzs", "--uzawasteps", type = int, default = [0], nargs = '*')
	parser.add_argument("-ne0", "--nepochs0", type = int, default = [200], nargs = '*')
	parser.add_argument("-ne1", "--nepochs1", type = int, default = [50], nargs = '*')
	parser.add_argument("-ne2", "--nepochs2", type = int, default = [10], nargs = '*')
	parser.add_argument("-nro", "--nrounds", type = int, default = [5], nargs = '*')
	parser.add_argument("-trs", "--trainsize", type = float, default = [None], nargs = '*')
	parser.add_argument("-vls", "--valsize", type = float, default = [None], nargs = '*')
	parser.add_argument("-tss", "--testsize", type = float, default = [None], nargs = '*')
	parser.add_argument("-see", "--seed", type = int, default = [None], nargs = '*')
	parser.add_argument("-exp", "--experiments", action = 'store_true')
	parser.add_argument("-avg", "--averageexperiments", action = 'store_true')
	args = parser.parse_args()

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	if args.experiments or args.averageexperiments:
		parameters = [(name, values) for name, values in vars(args).items() if name not in ['experiments', 'averageexperiments']]
		experiments(parameters, args.averageexperiments)
	else :
		parameters = [values[0] for name, values in vars(args).items() if name not in ['experiments', 'averageexperiments']]
		experiment(*parameters, False)



