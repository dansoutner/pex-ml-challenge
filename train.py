#!/usr/bin/env python
# -*- coding: utf-8 -*-


from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainer import links as L
import chainer
import numpy as np
import argparse
import os

from VGGnet import VGGNet
from VGGnet import VGGNetsmall
from dataset import PreprocessOnTheFlyDataset


def train(dataset_train_file, dataset_dev_file, model, mean_image_file=None, batchsize=32, gpu_id=0, max_epoch=20, img_size=224):

	# 1. Dataset and mean image
	if os.path.exists(mean_image_file):
		mean = np.load(mean_image_file)
	else:
		mean = np.ones((3, img_size, img_size)) * 128
	dataset_train = PreprocessOnTheFlyDataset(dataset_train_file, mean, img_size, random_flip=True, random_crop=True)
	dataset_dev = PreprocessOnTheFlyDataset(dataset_dev_file, mean, img_size, random_flip=False, random_crop=True)

	# 2. Iterator
	train_iter = iterators.SerialIterator(dataset_train, batchsize, shuffle=True)
	dev_iter = iterators.SerialIterator(dataset_dev, batchsize, False, False)

	# 3. Model
	with chainer.using_config('use_cudnn', 'never'):
		model = L.Classifier(model)
		if gpu_id >= 0:
			model.to_gpu(gpu_id)

		# 4. Optimizer
		optimizer = optimizers.Adam()
		optimizer.setup(model)

		# 5. Updater
		updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

		# 6. Trainer
		trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='{}_result'.format(model.__class__.__name__))

		# 7. Evaluator
		class TestModeEvaluator(extensions.Evaluator):

			def evaluate(self):
				model = self.get_target('main')
				ret = super(TestModeEvaluator, self).evaluate()
				return ret

		trainer.extend(extensions.LogReport())
		trainer.extend(TestModeEvaluator(dev_iter, model, device=gpu_id), trigger=(1, "epoch"))
		trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
		trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
		trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
		trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
		trainer.run()
		del trainer

		return model


def main():
	parser = argparse.ArgumentParser(description='Train model on input ')
	parser.add_argument('dataset_train', help='Path to training image-label list file')
	parser.add_argument('dataset_dev', help='Path to validation image-label list file')
	parser.add_argument('--mean-image', help='File with precomputed mean image')
	parser.add_argument('--batch-size', default=32, type=int,
						help='Set batch size')
	parser.add_argument('--gpu-id', default=-1, type=int,
						help='Set GPU id for computation, for CPU set -1')
	parser.add_argument('--max-epoch', default=20, type=int,
						help='Set maximum of computation epochs')

	args = parser.parse_args()

	model = VGGNetsmall()

	train(args.dataset_train, args.dataset_dev, model, mean_image_file=args.mean_image,
	      batchsize=args.batch_size, gpu_id=args.gpu_id, max_epoch=args.max_epoch)

if __name__ == "__main__":
	main()
